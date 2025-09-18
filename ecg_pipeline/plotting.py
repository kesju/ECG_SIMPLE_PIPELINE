"""Plotting helpers for ECG data."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .index_map import Segment

COLOR_MAP = {
    "gaps": "#999999",
    "outliers": "#d62728",
    "rdropouts": "#1f77b4",
}


def _segment_slices(segments: Iterable[Segment], total_points: int | None) -> Iterable[Tuple[int, int]]:
    for start, end in segments:
        if total_points is not None and start >= total_points:
            continue
        stop = end if total_points is None else min(end, total_points)
        if stop <= start:
            continue
        yield start, stop


def plot_ecg_with_annotations(
    signal: np.ndarray,
    sample_rate: int,
    annotations: Dict[str, Iterable[Segment]],
    *,
    total_points: int | None = None,
    title: str = "ECG with annotations",
    output_path: Path | None = None,
    show: bool = True,
) -> Path | None:
    """Plot the ECG signal and overlay annotated segments."""

    # Resolve total number of points as an int to satisfy type checkers
    n_points = min(total_points, signal.shape[0]) if total_points is not None else signal.shape[0]

    times = np.arange(n_points, dtype=np.float32) / float(sample_rate)
    values = signal[:n_points]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, values, color="black", linewidth=1.0, label="ECG start")

    # ...existing code...
    for label, segments in annotations.items():
        color = COLOR_MAP.get(label, None)
        for start, end in _segment_slices(segments, n_points):
            if start is None or end is None:
                continue  # Skip invalid segments
            start_time = times[start]
            end_index = min(end - 1, n_points - 1)
            end_time = times[end_index]
            if end_time <= start_time:
                end_time = start_time + (1.0 / float(sample_rate))
            ax.axvspan(start_time, end_time, color=color, alpha=0.25, label=label)
    # ...existing code...
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    output: Path | None = None
    if output_path is not None:
        output = Path(output_path)
        fig.savefig(output, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output

