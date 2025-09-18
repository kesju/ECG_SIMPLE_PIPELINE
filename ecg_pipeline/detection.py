"""Mock detection routines for outliers and R-dropouts."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .index_map import Segment, _merge_segments


def _iter_groups(mask: np.ndarray) -> Iterable[Tuple[int, int]]:
    """Yield contiguous (start, end) pairs where mask equals True."""

    if mask.size == 0:
        return
    diff = np.diff(mask.astype(int))
    starts = np.flatnonzero(np.concatenate(([mask[0]], diff == 1)))
    stops = np.flatnonzero(np.concatenate((diff == -1, [mask[-1]]))) + 1
    for start, stop in zip(starts, stops):
        yield int(start), int(stop)


def detect_outliers(
    signal: np.ndarray,
    amplitude_threshold: float,
    min_segment_length: int,
) -> List[Segment]:
    """Detect unusual amplitude spikes as proxy for outliers."""

    amplitude = np.abs(signal)
    mask = amplitude >= amplitude_threshold
    raw_segments = ((s, e) for s, e in _iter_groups(mask) if (e - s) >= min_segment_length)
    return _merge_segments(raw_segments)


def detect_r_dropouts(
    signal: np.ndarray,
    std_threshold: float,
    window_size: int,
    min_segment_length: int,
) -> List[Segment]:
    """Detect flat segments with low standard deviation as mock R-dropouts."""

    if signal.size == 0:
        return []

    window_size = max(1, int(window_size))
    if window_size > signal.size:
        window_size = signal.size

    # Rolling window std; use cumulative sums for memory efficiency.
    padded = np.pad(signal.astype(float), (window_size - 1, 0), mode="edge")
    cumsum = np.cumsum(padded)
    cumsum_sq = np.cumsum(padded**2)
    window_count = window_size
    totals = cumsum[window_size:] - cumsum[:-window_size]
    totals_sq = cumsum_sq[window_size:] - cumsum_sq[:-window_size]
    mean = totals / window_count
    variance = (totals_sq / window_count) - mean**2
    variance = np.clip(variance, a_min=0.0, a_max=None)
    rolling_std = np.sqrt(variance)

    # Align std array with signal length by padding the front.
    std_full = np.concatenate((np.full(window_size - 1, rolling_std[0]), rolling_std))

    mask = std_full <= std_threshold
    raw_segments = ((s, e) for s, e in _iter_groups(mask) if (e - s) >= min_segment_length)
    return _merge_segments(raw_segments)

