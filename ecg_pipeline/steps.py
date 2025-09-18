"""Processing primitives used by the denoising pipeline."""
from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

from .index_map import Segment, _merge_segments


def normalize_segments(segments: Iterable[Segment], length: int) -> List[Segment]:
    """Clip, sort, and merge segments within the signal length."""

    cleaned: List[Segment] = []
    for start, end in segments:
        if end <= start:
            continue
        start = max(0, int(start))
        end = min(length, int(end))
        if start < end:
            cleaned.append((start, end))
    return _merge_segments(cleaned)


def remove_segments(signal: np.ndarray, segments: Iterable[Segment]) -> Tuple[np.ndarray, np.ndarray]:
    """Remove segments from the signal and return the cleaned signal and kept indices."""

    length = signal.shape[0]
    normalized = normalize_segments(segments, length)
    if not normalized:
        kept_indices = np.arange(length, dtype=np.int64)
        return signal.copy(), kept_indices

    mask = np.ones(length, dtype=bool)
    for start, end in normalized:
        mask[start:end] = False

    kept_indices = np.nonzero(mask)[0]
    cleaned = signal[mask].copy()
    return cleaned, kept_indices


def remove_segments_in_place(buffer: np.ndarray, segments: Iterable[Segment]) -> Tuple[np.ndarray, np.ndarray]:
    """Remove segments without copying when possible by returning a new view."""

    cleaned, kept = remove_segments(buffer, segments)
    return cleaned, kept

