import matplotlib

matplotlib.use("Agg")

import os
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Sequence[str],
    title: str,
    save_path: str,
    cmap: str = "Blues",
    figsize=(6, 5),
) -> None:
    """
    Plot and save a single confusion matrix heatmap.

    This is a lightweight wrapper around matplotlib's imshow that standardises
    the look-and-feel across experiments.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    # Annotate cells with counts
    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > thresh else "black",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_confusion_grid(
    cms: Sequence[np.ndarray],
    labels: Sequence[str],
    row_titles: Sequence[str],
    col_titles: Sequence[str],
    save_path: str,
    cmap: str = "Blues",
    figsize: tuple[float, float] | None = None,
) -> None:
    """
    Plot a grid of confusion matrices (heatmap grid).

    Args:
        cms: Flat sequence of confusion matrices in row-major order. Length
            must be len(row_titles) * len(col_titles).
        labels: Shared label names for all confusion matrices.
        row_titles: Titles for each row (e.g., different frequency bands).
        col_titles: Titles for each column (e.g., time vs frequency).
        save_path: Where to save the resulting figure.
    """
    n_rows = len(row_titles)
    n_cols = len(col_titles)
    expected_len = n_rows * n_cols
    if len(cms) != expected_len:
        raise ValueError(
            f"Expected {expected_len} confusion matrices, got {len(cms)} instead."
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # If no explicit figsize provided, scale with grid size
    if figsize is None:
        # 3x3 inches per subplot as a heuristic
        figsize = (3.0 * n_cols, 3.0 * n_rows)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        squeeze=False,
    )

    vmin = min(cm.min() for cm in cms) if cms else 0.0
    vmax = max(cm.max() for cm in cms) if cms else 1.0

    for idx, cm in enumerate(cms):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]

        im = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)

        if row == 0:
            ax.set_title(col_titles[col], fontsize=10)

        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        if row == n_rows - 1:
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        else:
            ax.set_xticklabels([])
        ax.set_yticklabels(labels if col == 0 else [], fontsize=8)

        # Optionally annotate only main diagonal/off-diagonals if grids get busy.
        thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                value = int(cm[i, j])
                ax.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > thresh else "black",
                    fontsize=6,
                )

    # Add row titles as separate text on the left, aligned to the centre of each row.
    if len(row_titles) == n_rows:
        for row in range(n_rows):
            bbox = axes[row][0].get_position()
            y_center = (bbox.y0 + bbox.y1) / 2.0
            fig.text(
                0.04,
                y_center,
                row_titles[row],
                va="center",
                ha="right",
                fontsize=9,
            )

    # Manually adjust layout to leave space on the right for a separate colorbar
    fig.subplots_adjust(
        left=0.18,
        right=0.9,
        bottom=0.15,
        top=0.95,
        wspace=0.3,
        hspace=0.4,
    )

    # Place colorbar completely outside the grid on the right.
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel("Count", rotation=270, labelpad=12)
    plt.savefig(save_path)
    plt.close(fig)

