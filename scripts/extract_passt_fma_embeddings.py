"""
Extract PaSST embeddings for the FMA / LMD dataset, using the same metadata
CSV as the CLAP pipeline, and save them as an NPZ file consumable by the
multi-model periodicity analysis.

Output NPZ keys:
    - 'embeddings': np.ndarray, shape (B, F_passt, T_passt)
    - 'genres':     np.ndarray[str], shape (B,)
    - 'file_paths': np.ndarray[str], shape (B,)

Optimizations (see CLI flags):
    - Optional WAV mirror (faster load than MP3) via scripts/convert_fma_metadata_audio_to_wav.py
    - .wav files use soundfile + optional resample (faster than librosa full pipeline)
    - Prefetch: background thread loads/decode while GPU runs the previous batch
    - Batched forward: hear21passt accepts (n_sounds, n_samples) with padding
    - CUDA: host tensors pinned + non_blocking H2D copy
    - Optional torch.compile on CUDA (PyTorch 2+)
    - Optional suppression of stdout inside model forward (library debug prints)

Multi-GPU: use data parallelism with --num_shards / --shard_index and merge NPZ via
scripts/merge_fma_embedding_npz_shards.py.

Resume: --resume_from_npz points to a prior NPZ (same metadata audio_path strings).
Paths are matched after abspath + normpath + normcase; cached rows are skipped on GPU.
Checkpoint: <output>.checkpoint.npz (or --checkpoint_path) is merged on startup; with
--checkpoint_every N it is rewritten every N newly extracted tracks so kills/timeouts
lose at most ~N tracks of progress.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import queue
import zipfile
import tempfile
import threading
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from hear21passt.base import get_timestamp_embeddings, load_model
from tqdm.auto import tqdm


METADATA_PATH = "data_artifacts/fma_metadata.csv"
OUTPUT_DIR = "data_artifacts"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "passt_embeddings_t64.npz")
SAMPLE_RATE = 32000

_SENTINEL = object()


def _norm_audio_path(p: str) -> str:
    """Stable key for resume matching (absolute, normalized); cwd-dependent for relative paths."""
    if not p or not isinstance(p, str):
        return ""
    return os.path.normcase(os.path.normpath(os.path.abspath(p)))


def infer_audio_path_base(metadata_path: str) -> str:
    """
    Directory used to resolve relative audio_path (and relative wav_strip_prefix) strings.

    If metadata lives under .../<repo>/data_artifacts/*.csv, relative paths in the CSV are
    almost always relative to <repo> (sibling ``data/``), not data_artifacts/.
    Otherwise use the directory containing the CSV.
    """
    d = os.path.dirname(os.path.abspath(metadata_path))
    if os.path.basename(d) == "data_artifacts":
        return os.path.dirname(d)
    return d


def _audio_path_key(p: str, audio_path_base: str) -> str:
    """Stable key for resume/checkpoint matching; relative paths are resolved under audio_path_base."""
    if not p or not isinstance(p, str):
        return ""
    p = p.strip()
    base = os.path.abspath(audio_path_base)
    if os.path.isabs(p):
        ap = os.path.abspath(p)
    else:
        ap = os.path.abspath(os.path.join(base, p))
    return os.path.normcase(os.path.normpath(ap))


def load_resume_embedding_map(
    resume_path: Optional[str],
    *,
    verbose: bool = True,
    audio_path_base: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Map normalized audio_path -> embedding array (F, T) from a prior NPZ.
    Later duplicates in file_paths overwrite earlier entries.
    """
    if not resume_path:
        return {}
    if not os.path.isfile(resume_path):
        if verbose:
            print(f"[extract_passt_fma] Resume file not found ({resume_path}); starting fresh.")
        return {}
    try:
        data = np.load(resume_path, allow_pickle=True)
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as e:
        # Always log: checkpoint loads use verbose=False but a corrupt file must be visible.
        print(
            f"[extract_passt_fma] Resume NPZ unreadable (empty/truncated/corrupt): "
            f"{resume_path} ({type(e).__name__}: {e}). Starting without it."
        )
        return {}
    if "embeddings" not in data or "file_paths" not in data:
        if verbose:
            print("[extract_passt_fma] Resume NPZ missing embeddings or file_paths; ignoring.")
        return {}
    emb = data["embeddings"]
    fps = data["file_paths"]
    out: Dict[str, np.ndarray] = {}
    dup = 0
    for i in range(len(fps)):
        raw = str(fps[i])
        k = (
            _audio_path_key(raw, audio_path_base)
            if audio_path_base
            else _norm_audio_path(raw)
        )
        if not k:
            continue
        if k in out:
            dup += 1
        out[k] = np.asarray(emb[i], dtype=np.float32).copy()
    if verbose:
        if dup:
            print(f"[extract_passt_fma] Resume: {dup} duplicate path(s) in NPZ; kept last.")
        print(f"[extract_passt_fma] Resume: {len(out)} track(s) from {resume_path}")
    return out


def _save_checkpoint_npz(
    embeddings_list: List[np.ndarray],
    file_paths: List[str],
    path: str,
) -> None:
    """Atomic write: padded embeddings + file_paths (genres omitted; filled at final save)."""
    if not embeddings_list or not path:
        return
    out_path = os.path.abspath(path)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    max_t = max(e.shape[1] for e in embeddings_list)
    feat_dim = embeddings_list[0].shape[0]
    padded = np.zeros((len(embeddings_list), feat_dim, max_t), dtype=np.float32)
    for i, e in enumerate(embeddings_list):
        padded[i, :, : e.shape[1]] = e

    fd, tmp_path = tempfile.mkstemp(suffix=".npz.tmp", dir=out_dir)
    os.close(fd)
    try:
        np.savez_compressed(
            tmp_path,
            embeddings=padded,
            file_paths=np.asarray(file_paths, dtype=object),
            genres=np.asarray([""] * len(file_paths), dtype=object),
        )
        os.replace(tmp_path, out_path)
    except BaseException:
        try:
            if os.path.isfile(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass
        raise
    print(
        f"[extract_passt_fma] Checkpoint: {len(file_paths)} new track(s) -> {out_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract PaSST embeddings for the FMA dataset."
    )
    parser.add_argument(
        "--metadata_path",
        default=METADATA_PATH,
        help=f"CSV containing at least audio_path and genre columns (default: {METADATA_PATH}).",
    )
    parser.add_argument(
        "--output_file",
        default=OUTPUT_FILE,
        help=f"Output NPZ path (default: {OUTPUT_FILE}).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to use, e.g. 'cuda', 'cuda:0', or 'cpu'. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--max_tracks",
        type=int,
        default=800,
        help="Optional cap on total tracks after stratified sampling. Use -1 for no limit.",
    )
    parser.add_argument(
        "--max_per_genre",
        type=int,
        default=100,
        help="Maximum tracks to keep per genre during stratified sampling.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help=(
            "Split the post-sampling track list into this many disjoint shards "
            "(data-parallel across GPUs). Use 1 for no sharding."
        ),
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Zero-based shard id; must satisfy 0 <= shard_index < num_shards.",
    )
    parser.add_argument(
        "--wav_mirror_root",
        default=None,
        help=(
            "If set, load WAV from this tree: "
            "join(wav_mirror_root, relpath(abspath(audio_path), wav_strip_prefix) "
            "with .mp3/.m4a etc replaced by .wav). Falls back to original path if WAV missing."
        ),
    )
    parser.add_argument(
        "--wav_strip_prefix",
        default=None,
        help="Required when --wav_mirror_root is set (see --wav_mirror_root).",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=2,
        help="Decoded-audio queue depth for CPU/GPU overlap. 0 disables the loader thread.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of tracks per PaSST forward (hear21passt pads within batch).",
    )
    parser.add_argument(
        "--silence_model_stdout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Redirect stdout during get_timestamp_embeddings (default: true).",
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Wrap model with torch.compile (PyTorch 2+, CUDA only; first batches may be slow).",
    )
    parser.add_argument(
        "--resume_from_npz",
        default=None,
        help=(
            "Existing NPZ with keys embeddings, file_paths: skip any audio_path that "
            "matches (normalized). Re-run merges cached rows with newly extracted ones "
            "in metadata order."
        ),
    )
    parser.add_argument(
        "--auto_resume",
        action="store_true",
        help=(
            "If --output_file already exists, use it as resume checkpoint "
            "(same as --resume_from_npz pointing at that file). Ignored if "
            "--resume_from_npz is set."
        ),
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=0,
        help=(
            "Write a crash-safe NPZ every N newly extracted tracks (0 = never). "
            "Default checkpoint path is OUTPUT.checkpoint.npz (see --checkpoint_path). "
            "On startup, that file is merged into resume if it exists."
        ),
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Checkpoint NPZ path (default: <output_file>.checkpoint.npz).",
    )
    parser.add_argument(
        "--audio_path_base",
        default=None,
        help=(
            "Directory against which relative audio_path (and relative wav_strip_prefix) "
            "are resolved. Default: parent of data_artifacts if metadata is under "
            "data_artifacts/, else directory of the metadata CSV."
        ),
    )
    return parser.parse_args()


def resolve_load_path(
    original_audio_path: str,
    wav_mirror_root: Optional[str],
    wav_strip_prefix: Optional[str],
    audio_path_base: Optional[str] = None,
) -> str:
    """Return path to open on disk (prefer mirrored WAV when present)."""
    def _abs_under_base(p: str) -> str:
        if not p:
            return p
        if os.path.isabs(p):
            return os.path.abspath(p)
        if audio_path_base:
            return os.path.abspath(os.path.join(os.path.abspath(audio_path_base), p))
        return os.path.abspath(p)

    if not wav_mirror_root:
        return _abs_under_base(original_audio_path)
    if not wav_strip_prefix:
        raise ValueError("wav_strip_prefix is required when wav_mirror_root is set")

    orig_abs = _abs_under_base(original_audio_path)
    strip_abs = _abs_under_base(wav_strip_prefix)
    try:
        rel = os.path.relpath(orig_abs, strip_abs)
    except ValueError:
        return original_audio_path

    base, _ = os.path.splitext(rel)
    wav_path = os.path.join(os.path.abspath(wav_mirror_root), base + ".wav")
    if os.path.isfile(wav_path):
        return wav_path
    return original_audio_path


@contextlib.contextmanager
def _maybe_silence_stdout(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as dn:
        with contextlib.redirect_stdout(dn):
            yield


def _load_audio_mono_32k(load_path: str) -> np.ndarray:
    """
    Fast path for WAV via soundfile; MP3/other via librosa.
    Returns float32 mono (N,) at SAMPLE_RATE.
    """
    ext = os.path.splitext(load_path)[1].lower()
    if ext == ".wav":
        data, sr = sf.read(load_path, dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1).astype(np.float32, copy=False)
        elif data.ndim != 1:
            raise ValueError(f"Unexpected WAV shape {data.shape} for {load_path}")
        if sr != SAMPLE_RATE:
            data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
            data = data.astype(np.float32, copy=False)
        return data
    waveform, _ = librosa.load(load_path, sr=SAMPLE_RATE, mono=True)
    return waveform.astype(np.float32, copy=False)


def _load_passt_model(device: str, torch_compile: bool = False) -> torch.nn.Module:
    print("[extract_passt_fma] Loading PaSST model (hear21passt.base)...")
    model = load_model()
    model.to(device)
    model.eval()
    if torch_compile and str(device).startswith("cuda"):
        try:
            model = torch.compile(model)  # type: ignore[assignment]
            print("[extract_passt_fma] torch.compile enabled")
        except Exception as e:
            print(f"[extract_passt_fma] torch.compile skipped: {e}")
    return model


def _forward_batch(
    waveforms: List[np.ndarray],
    model: torch.nn.Module,
    device: str,
    silence_stdout: bool,
) -> List[np.ndarray]:
    """
    waveforms: list of 1D float mono samples.
    Returns list of (D, T_frames) numpy arrays.
    """
    if not waveforms:
        return []
    max_len = max(w.shape[0] for w in waveforms)
    b = len(waveforms)
    x = np.zeros((b, max_len), dtype=np.float32)
    for i, w in enumerate(waveforms):
        x[i, : w.shape[0]] = w
    if str(device).startswith("cuda"):
        audio_tensor = torch.from_numpy(x).pin_memory().to(device, non_blocking=True)
    else:
        audio_tensor = torch.from_numpy(x).to(device)

    with torch.no_grad():
        with _maybe_silence_stdout(silence_stdout):
            emb, _ = get_timestamp_embeddings(audio_tensor, model)

    emb = emb.cpu().numpy()
    out: List[np.ndarray] = []
    for i in range(b):
        out.append(emb[i].T.astype(np.float32))
    return out


def _producer_loop(
    sampled_df: pd.DataFrame,
    wav_mirror_root: Optional[str],
    wav_strip_prefix: Optional[str],
    audio_path_base: Optional[str],
    q: "queue.Queue",
) -> None:
    try:
        for _, row in sampled_df.iterrows():
            orig_path = row.get("audio_path")
            genre = str(row.get("genre", ""))
            if not orig_path or not isinstance(orig_path, str):
                q.put(("skip", None, None, None))
                continue
            load_path = resolve_load_path(
                orig_path, wav_mirror_root, wav_strip_prefix, audio_path_base
            )
            if not os.path.exists(load_path):
                q.put(("skip", None, None, None))
                continue
            try:
                waveform = _load_audio_mono_32k(load_path)
                q.put(("ok", orig_path, genre, waveform))
            except Exception as e:
                print(f"[extract_passt_fma] Error loading {load_path}: {e}")
                q.put(("skip", None, None, None))
    finally:
        q.put(_SENTINEL)


def _run_extraction_core(
    sampled_df: pd.DataFrame,
    model: torch.nn.Module,
    device: str,
    wav_mirror_root: Optional[str],
    wav_strip_prefix: Optional[str],
    audio_path_base: Optional[str],
    prefetch: int,
    batch_size: int,
    silence_model_stdout: bool,
    tqdm_desc: str = "PaSST FMA (new tracks only)",
    checkpoint_every: int = 0,
    checkpoint_path: Optional[str] = None,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Run loader + PaSST on every row of sampled_df (rows must be extractable).
    Returns parallel lists: embeddings (each F x T), file_paths (metadata strings).
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    embeddings_list: List[np.ndarray] = []
    file_paths: List[str] = []
    ckpt_counter = 0

    n_rows = len(sampled_df)
    print(
        f"[extract_passt_fma] Extracting {n_rows} tracks (batch_size={batch_size}, "
        f"prefetch={prefetch})..."
    )

    if prefetch > 0:
        q: queue.Queue = queue.Queue(maxsize=max(prefetch, batch_size * 2))
        thread = threading.Thread(
            target=_producer_loop,
            args=(sampled_df, wav_mirror_root, wav_strip_prefix, audio_path_base, q),
            daemon=True,
        )
        thread.start()
        source_iter = None
    else:
        q = None
        source_iter = sampled_df.iterrows()

    pbar = tqdm(total=n_rows, desc=tqdm_desc)

    batch_wavs: List[np.ndarray] = []
    batch_orig: List[str] = []
    batch_genres: List[str] = []

    def _maybe_checkpoint() -> None:
        nonlocal ckpt_counter
        if checkpoint_every <= 0 or not checkpoint_path:
            return
        ckpt_counter += 1
        if ckpt_counter % checkpoint_every != 0:
            return
        try:
            _save_checkpoint_npz(embeddings_list, file_paths, checkpoint_path)
        except OSError as e:
            print(f"[extract_passt_fma] Checkpoint save failed (continuing extract): {e}")

    def flush_batch() -> None:
        nonlocal batch_wavs, batch_orig, batch_genres
        if not batch_wavs:
            return
        try:
            embs = _forward_batch(
                batch_wavs, model=model, device=device, silence_stdout=silence_model_stdout
            )
            for orig, e in zip(batch_orig, embs):
                embeddings_list.append(e)
                file_paths.append(orig)
                _maybe_checkpoint()
        except Exception as e:
            print(f"[extract_passt_fma] Batch forward failed ({e}); falling back to one-by-one.")
            for w, orig in zip(batch_wavs, batch_orig):
                try:
                    one = _forward_batch(
                        [w], model=model, device=device, silence_stdout=silence_model_stdout
                    )[0]
                    embeddings_list.append(one)
                    file_paths.append(orig)
                    _maybe_checkpoint()
                except Exception as e2:
                    print(f"[extract_passt_fma] Error processing {orig}: {e2}")
        batch_wavs = []
        batch_orig = []
        batch_genres = []

    rows_done = 0
    while True:
        if prefetch > 0:
            item = q.get()
            if item is _SENTINEL:
                flush_batch()
                pbar.update(n_rows - rows_done)
                break
            status, orig_path, genre, waveform = item
            rows_done += 1
            pbar.update(1)
            if status != "ok" or waveform is None:
                continue
            batch_wavs.append(waveform)
            batch_orig.append(orig_path)
            batch_genres.append(genre)
            if len(batch_wavs) >= batch_size:
                flush_batch()
        else:
            try:
                _, row = next(source_iter)
            except StopIteration:
                flush_batch()
                break
            rows_done += 1
            pbar.update(1)
            orig_path = row.get("audio_path")
            genre = str(row.get("genre", ""))
            if not orig_path or not isinstance(orig_path, str):
                continue
            load_path = resolve_load_path(
                orig_path, wav_mirror_root, wav_strip_prefix, audio_path_base
            )
            if not os.path.exists(load_path):
                continue
            try:
                waveform = _load_audio_mono_32k(load_path)
            except Exception as e:
                print(f"[extract_passt_fma] Error loading {load_path}: {e}")
                continue
            batch_wavs.append(waveform)
            batch_orig.append(orig_path)
            batch_genres.append(genre)
            if len(batch_wavs) >= batch_size:
                flush_batch()

    pbar.close()

    if prefetch > 0:
        thread.join(timeout=1.0)

    if (
        checkpoint_every > 0
        and checkpoint_path
        and embeddings_list
        and ckpt_counter % checkpoint_every != 0
    ):
        _save_checkpoint_npz(embeddings_list, file_paths, checkpoint_path)

    return embeddings_list, file_paths


def run_extraction_on_dataframe(
    sampled_df: pd.DataFrame,
    device: str,
    audio_path_base: str,
    wav_mirror_root: Optional[str] = None,
    wav_strip_prefix: Optional[str] = None,
    prefetch: int = 2,
    batch_size: int = 4,
    silence_model_stdout: bool = True,
    torch_compile: bool = False,
    resume_from_npz: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    checkpoint_every: int = 0,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Run PaSST forward where needed; merge with optional resume NPZ in metadata order."""
    resume_map = load_resume_embedding_map(
        resume_from_npz, audio_path_base=audio_path_base
    )
    if checkpoint_path and os.path.isfile(checkpoint_path):
        ck_map = load_resume_embedding_map(
            checkpoint_path, verbose=False, audio_path_base=audio_path_base
        )
        if ck_map:
            for k, v in ck_map.items():
                resume_map[k] = v
            print(
                f"[extract_passt_fma] Checkpoint merge: {len(ck_map)} path(s) from "
                f"{checkpoint_path}"
            )
        else:
            sz = os.path.getsize(checkpoint_path)
            print(
                f"[extract_passt_fma] Checkpoint file exists ({sz} bytes) but loaded "
                f"0 usable entries from {checkpoint_path} (corrupt/empty, or unreadable; "
                f"see messages above). Relative paths must match --audio_path_base "
                f"({audio_path_base!r})."
            )

    rows_to_extract: List[pd.Series] = []
    n_resume = 0
    for _, row in sampled_df.iterrows():
        orig = row.get("audio_path")
        if not orig or not isinstance(orig, str):
            continue
        key = _audio_path_key(orig, audio_path_base)
        if key in resume_map:
            n_resume += 1
            continue
        load_path = resolve_load_path(
            orig, wav_mirror_root, wav_strip_prefix, audio_path_base
        )
        if not os.path.exists(load_path):
            continue
        rows_to_extract.append(row)

    sub_df = pd.DataFrame(rows_to_extract) if rows_to_extract else pd.DataFrame()
    n_need = len(sub_df)
    print(
        f"[extract_passt_fma] This run: {len(sampled_df)} rows in shard/metadata slice, "
        f"{n_resume} skipped (already in resume NPZ), {n_need} will go through GPU loader."
    )
    if len(resume_map) > 0 and n_resume == 0:
        ex_npz = next(iter(resume_map.keys()), "")
        ex_csv = ""
        for _, r in sampled_df.iterrows():
            p = r.get("audio_path")
            if p and isinstance(p, str):
                ex_csv = _audio_path_key(p, audio_path_base)
                break
        print(
            "[extract_passt_fma] WARNING: resume NPZ has "
            f"{len(resume_map)} path(s) but 0 matched this job's CSV (wrong shard file, "
            f"wrong --audio_path_base, or path strings differ). Example CSV key: {ex_csv!r} "
            f"vs NPZ key: {ex_npz!r}"
        )

    new_by_key: Dict[str, np.ndarray] = {}
    if n_need > 0:
        model = _load_passt_model(device, torch_compile=torch_compile)
        new_embs, new_fps = _run_extraction_core(
            sub_df,
            model=model,
            device=device,
            wav_mirror_root=wav_mirror_root,
            wav_strip_prefix=wav_strip_prefix,
            audio_path_base=audio_path_base,
            prefetch=prefetch,
            batch_size=batch_size,
            silence_model_stdout=silence_model_stdout,
            tqdm_desc=f"PaSST new only ({n_need} tracks, resume skips already done)",
            checkpoint_every=checkpoint_every,
            checkpoint_path=checkpoint_path if checkpoint_every > 0 else None,
        )
        for emb, fp in zip(new_embs, new_fps):
            new_by_key[_audio_path_key(fp, audio_path_base)] = emb

    embeddings_list: List[np.ndarray] = []
    genres: List[str] = []
    file_paths: List[str] = []
    n_assembled_resume = 0
    n_assembled_new = 0

    for _, row in sampled_df.iterrows():
        orig = row.get("audio_path")
        genre = str(row.get("genre", ""))
        if not orig or not isinstance(orig, str):
            continue
        key = _audio_path_key(orig, audio_path_base)
        load_path = resolve_load_path(
            orig, wav_mirror_root, wav_strip_prefix, audio_path_base
        )
        if key in resume_map:
            embeddings_list.append(resume_map[key])
            genres.append(genre)
            file_paths.append(orig)
            n_assembled_resume += 1
        elif key in new_by_key:
            embeddings_list.append(new_by_key[key])
            genres.append(genre)
            file_paths.append(orig)
            n_assembled_new += 1
        else:
            if not os.path.exists(load_path):
                continue
            print(
                f"[extract_passt_fma] Warning: no embedding for {orig} "
                "(missing from resume and extraction failed or skipped)."
            )

    if not embeddings_list:
        raise RuntimeError("[extract_passt_fma] No embeddings were extracted.")

    print(
        f"[extract_passt_fma] Assembled {len(embeddings_list)} tracks: "
        f"{n_assembled_resume} from resume NPZ, {n_assembled_new} newly computed."
    )

    max_T = max(emb.shape[1] for emb in embeddings_list)
    feat_dim = embeddings_list[0].shape[0]

    padded = np.zeros((len(embeddings_list), feat_dim, max_T), dtype=np.float32)
    for i, emb in enumerate(embeddings_list):
        t_i = emb.shape[1]
        padded[i, :, :t_i] = emb

    return padded, genres, file_paths


def extract_passt_fma_embeddings(
    metadata_df: pd.DataFrame,
    device: str,
    audio_path_base: str,
    max_tracks: Optional[int] = 800,
    max_per_genre: int = 100,
    num_shards: int = 1,
    shard_index: int = 0,
    wav_mirror_root: Optional[str] = None,
    wav_strip_prefix: Optional[str] = None,
    prefetch: int = 2,
    batch_size: int = 4,
    silence_model_stdout: bool = True,
    torch_compile: bool = False,
    resume_from_npz: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    checkpoint_every: int = 0,
) -> Tuple[np.ndarray, List[str], List[str]]:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not (0 <= shard_index < num_shards):
        raise ValueError(
            f"shard_index must be in [0, num_shards), got {shard_index=} {num_shards=}"
        )

    sampled_df = build_stratified_sample(
        metadata_df,
        max_tracks=max_tracks,
        max_per_genre=max_per_genre,
    )
    if num_shards > 1:
        n_before = len(sampled_df)
        sampled_df = sampled_df.iloc[shard_index::num_shards].reset_index(drop=True)
        print(
            f"[extract_passt_fma] Shard {shard_index}/{num_shards}: "
            f"{len(sampled_df)} / {n_before} tracks after split "
            f"(each GPU runs a disjoint slice; e.g. two shards ≈ half each, not duplicate 17k+17k)."
        )

    return run_extraction_on_dataframe(
        sampled_df,
        device=device,
        audio_path_base=audio_path_base,
        wav_mirror_root=wav_mirror_root,
        wav_strip_prefix=wav_strip_prefix,
        prefetch=prefetch,
        batch_size=batch_size,
        silence_model_stdout=silence_model_stdout,
        torch_compile=torch_compile,
        resume_from_npz=resume_from_npz,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
    )


def build_stratified_sample(
    metadata_df: pd.DataFrame,
    max_tracks: Optional[int],
    max_per_genre: int,
) -> pd.DataFrame:
    if "genre" not in metadata_df.columns:
        raise ValueError(
            f"Metadata CSV must contain 'genre' column for stratified sampling, "
            f"got columns: {list(metadata_df.columns)}"
        )

    sampled_indices: List[int] = []
    for _, group in metadata_df.groupby("genre"):
        idx = group.index.to_list()
        if len(idx) > max_per_genre:
            idx = idx[:max_per_genre]
        sampled_indices.extend(idx)
    sampled_indices = sorted(sampled_indices)
    if max_tracks is not None and len(sampled_indices) > max_tracks:
        sampled_indices = sampled_indices[:max_tracks]

    return metadata_df.loc[sampled_indices].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    output_dir = os.path.dirname(args.output_file) or "."
    os.makedirs(output_dir, exist_ok=True)

    if args.wav_mirror_root and not args.wav_strip_prefix:
        raise SystemExit(
            "When using --wav_mirror_root you must also set --wav_strip_prefix "
            "(same values as for convert_fma_metadata_audio_to_wav.py)."
        )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[extract_passt_fma] Using device: {device}")

    metadata_path = args.metadata_path
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata CSV not found at {metadata_path}. "
            "Please run the FMA data preparation first."
        )

    print(f"[extract_passt_fma] Loading metadata from {metadata_path}...")
    metadata = pd.read_csv(metadata_path)
    audio_path_base = (
        os.path.abspath(args.audio_path_base)
        if args.audio_path_base
        else infer_audio_path_base(metadata_path)
    )
    print(f"[extract_passt_fma] audio_path_base={audio_path_base}")
    if "audio_path" not in metadata.columns or "genre" not in metadata.columns:
        raise ValueError(
            f"Metadata CSV must contain 'audio_path' and 'genre' columns, "
            f"got columns: {list(metadata.columns)}"
        )

    if args.resume_from_npz:
        resume_path: Optional[str] = args.resume_from_npz
        print(f"[extract_passt_fma] Resume: {resume_path}")
    elif args.auto_resume and os.path.isfile(args.output_file):
        resume_path = args.output_file
        print(f"[extract_passt_fma] --auto_resume: reusing existing {resume_path}")
    elif args.auto_resume:
        resume_path = None
        print(
            "[extract_passt_fma] --auto_resume: output file not found yet; "
            "full extraction (no skip)."
        )
    else:
        resume_path = None
        print(
            "[extract_passt_fma] Resume: off. To skip already-done tracks, use "
            "--resume_from_npz <path> or --auto_resume (when output exists)."
        )

    ckpt_path = args.checkpoint_path or (args.output_file + ".checkpoint.npz")
    if args.checkpoint_every > 0:
        print(
            f"[extract_passt_fma] Checkpoint every {args.checkpoint_every} new track(s) -> {ckpt_path}"
        )

    max_tracks = None if args.max_tracks is not None and args.max_tracks < 0 else args.max_tracks
    embeddings, genres, file_paths = extract_passt_fma_embeddings(
        metadata_df=metadata,
        device=device,
        audio_path_base=audio_path_base,
        max_tracks=max_tracks,
        max_per_genre=args.max_per_genre,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        wav_mirror_root=args.wav_mirror_root,
        wav_strip_prefix=args.wav_strip_prefix,
        prefetch=args.prefetch,
        batch_size=args.batch_size,
        silence_model_stdout=args.silence_model_stdout,
        torch_compile=args.torch_compile,
        resume_from_npz=resume_path,
        checkpoint_path=ckpt_path,
        checkpoint_every=args.checkpoint_every,
    )

    print(
        f"[extract_passt_fma] Final embeddings shape: {embeddings.shape} "
        f"(B={embeddings.shape[0]}, F={embeddings.shape[1]}, T={embeddings.shape[2]})"
    )

    print(f"[extract_passt_fma] Saving NPZ to {args.output_file}...")
    np.savez_compressed(
        args.output_file,
        embeddings=embeddings,
        genres=np.asarray(genres),
        file_paths=np.asarray(file_paths),
    )
    if os.path.isfile(ckpt_path):
        try:
            os.remove(ckpt_path)
            print(f"[extract_passt_fma] Removed checkpoint after successful save: {ckpt_path}")
        except OSError as e:
            print(f"[extract_passt_fma] Could not remove checkpoint {ckpt_path}: {e}")
    print("[extract_passt_fma] Done.")


if __name__ == "__main__":
    main()
