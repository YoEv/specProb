import os
import glob
import jams
import pandas as pd
import tqdm

def scan_files(audio_dir, annot_dir):
    """
    Scans for audio files matching *_{BN,SS}*_*_comp_mix.wav
    and finds corresponding JAMS files.
    """
    # Glob patterns
    # Simplified patterns to avoid issues with underscores
    bn_pattern = os.path.join(audio_dir, "*BN*comp_mix.wav")
    ss_pattern = os.path.join(audio_dir, "*SS*comp_mix.wav")
    
    print(f"Scanning {audio_dir}...")
    print(f"Pattern BN: {bn_pattern}")
    print(f"Pattern SS: {ss_pattern}")
    
    audio_files = glob.glob(bn_pattern) + glob.glob(ss_pattern)
    audio_files.sort()
    
    pairs = []
    
    print(f"Found {len(audio_files)} audio files matching BN/SS comp criteria.")
    
    for audio_path in audio_files:
        basename = os.path.basename(audio_path)
        # Audio: 00_BN1-129-Eb_comp_mix.wav
        # Annot: 00_BN1-129-Eb_comp.jams
        # Remove _mix.wav and add .jams
        annot_basename = basename.replace("_mix.wav", ".jams")
        annot_path = os.path.join(annot_dir, annot_basename)
        
        if os.path.exists(annot_path):
            pairs.append((audio_path, annot_path))
        else:
            print(f"Warning: Annotation not found for {basename}")
            
    print(f"Matched {len(pairs)} pairs.")
    return pairs

def simplify_chord(chord_label):
    """
    Simplifies a JAMS chord label to Root:Quality.
    Supported qualities: maj, min.
    Everything else -> N.
    """
    if chord_label == 'N':
        return 'N'
        
    try:
        root, quality = chord_label.split(':')
    except ValueError:
        # If split fails (e.g. just "C" or complex string), map to N
        return 'N'
        
    # Simplify quality
    # Check if it starts with maj or min
    if quality == 'maj' or quality.startswith('maj'):
        return f"{root}:maj"
    elif quality == 'min' or quality.startswith('min'):
        return f"{root}:min"
    
    # Common extensions mapping
    # 7 -> maj (dominant 7 often serves as major in simple triad classification, or maybe N?)
    # Let's strictly follow plan: "keep: maj, min. drop or map others -> N"
    # Wait, the plan example says: "C:min7 -> C:min".
    # So if it contains 'min', it is min.
    
    # What about '7'? Usually dominant. Is it major?
    # In triad classification, 7 implies major third.
    # But strictly speaking, if we want high confidence, maybe N?
    # Let's try to map '7' to 'maj' for now, as it has a major third.
    
    if quality == '7':
        return f"{root}:maj"
        
    # sus, dim, aug -> N
    
    return 'N'

def parse_jams_chords(jams_path):
    jam = jams.load(jams_path)
    
    # GuitarSet usually has chord annotations in 'chord' namespace
    anns = jam.search(namespace='chord')
    if not anns:
        return []
    
    # Use the first chord annotation found
    ann = anns[0]
    
    segments = []
    for obs in ann.data:
        t_start = obs.time
        duration = obs.duration
        t_end = t_start + duration
        value = obs.value
        
        simple_chord = simplify_chord(value)
        
        segments.append({
            't_start': t_start,
            't_end': t_end,
            'chord_label': simple_chord,
            'original_chord': value
        })
        
    return segments

def create_segments(pairs, output_path):
    all_segments = []
    
    for audio_path, annot_path in tqdm.tqdm(pairs, desc="Processing files"):
        basename = os.path.basename(audio_path)
        # Extract track_id from basename (e.g., 00_BN1-129-Eb_comp)
        track_id = basename.replace("_mix.wav", "")
        
        segments = parse_jams_chords(annot_path)
        
        for seg in segments:
            seg['track_id'] = track_id
            seg['audio_path'] = audio_path
            all_segments.append(seg)
            
    df = pd.DataFrame(all_segments)
    
    # Filter out N? Plan says "12 roots x {major, minor} + N (optional)".
    # Let's keep N for now, we can filter later or use it as a class.
    
    print(f"Total segments: {len(df)}")
    print("Chord distribution:")
    print(df['chord_label'].value_counts())
    
    df.to_csv(output_path, index=False)
    print(f"Saved segments to {output_path}")

if __name__ == "__main__":
    AUDIO_DIR = "/home/evev/specProb/guitarSet/audio_mono-pickup_mix"
    ANNOT_DIR = "/home/evev/specProb/guitarSet/annotation"
    OUTPUT_CSV = "/home/evev/specProb/data_artifacts/segments.csv"
    
    pairs = scan_files(AUDIO_DIR, ANNOT_DIR)
    create_segments(pairs, OUTPUT_CSV)
