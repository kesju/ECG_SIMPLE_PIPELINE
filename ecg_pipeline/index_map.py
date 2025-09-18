"""Utility for tracking index mappings between processing stages."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

Segment = Tuple[int, int]


def _merge_segments(segments: Iterable[Segment]) -> List[Segment]:
    """Merge overlapping or adjacent segments."""

    sorted_segments = sorted((start, end) for start, end in segments if end > start)
    if not sorted_segments:
        return []

    merged: List[Segment] = []
    cur_start, cur_end = sorted_segments[0]
    for start, end in sorted_segments[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    return merged


@dataclass(frozen=True)
class StageMapping:
    """Mapping metadata between two stages."""

    child: str
    parent: str
    kept_indices: Sequence[int]


class IndexMap:
    """Store removal metadata and project indices across stages."""

    def __init__(self) -> None:
        self._mappings: Dict[str, StageMapping] = {}
        self._segments: Dict[str, Tuple[str, List[Segment]]] = {}

    def add_mapping(self, child: str, parent: str, kept_indices: Sequence[int]) -> None:
        """Record how indices from *child* map back to *parent*."""

        self._mappings[child] = StageMapping(child=child, parent=parent, kept_indices=tuple(kept_indices))

    def record_segments(self, label: str, stage: str, segments: Sequence[Segment]) -> None:
        """Store detected segments for later projection."""

        merged = _merge_segments(segments)
        self._segments[label] = (stage, merged)

    def get_segments(self, label: str) -> List[Segment]:
        """Return stored segments without projection."""

        stage, segments = self._segments[label]
        return list(segments)

    def project(self, label: str, target_stage: str) -> List[Segment]:
        """Project stored segments for *label* to the *target_stage* coordinate space."""

        if label not in self._segments:
            raise KeyError(f"Unknown label: {label}")

        stage, segments = self._segments[label]
        if stage == target_stage:
            return list(segments)

        projected: List[Segment] = list(segments)
        current_stage = stage
        while current_stage != target_stage:
            mapping = self._mappings.get(current_stage)
            if mapping is None:
                raise ValueError(
                    f"Cannot project from {stage} to {target_stage}: missing mapping for {current_stage}."
                )
            projected = self._project_segments_once(projected, mapping.kept_indices)
            current_stage = mapping.parent

        return _merge_segments(projected)

    @staticmethod
    def _project_segments_once(segments: Sequence[Segment], kept_indices: Sequence[int]) -> List[Segment]:
        """Project segments from child stage to its parent using *kept_indices*."""

        if not segments:
            return []
        if not kept_indices:
            return []

        parent_segments: List[Segment] = []
        for start, end in segments:
            if start >= end:
                continue
            start = max(start, 0)
            end = min(end, len(kept_indices))
            if start >= end:
                continue
            parent_idxs = [kept_indices[i] for i in range(start, end)]
            if not parent_idxs:
                continue

            # Group consecutive parent indices into segments
            run_start = parent_idxs[0]
            prev_idx = parent_idxs[0]
            for parent_idx in parent_idxs[1:]:
                if parent_idx != prev_idx + 1:
                    parent_segments.append((run_start, prev_idx + 1))
                    run_start = parent_idx
                prev_idx = parent_idx
            parent_segments.append((run_start, prev_idx + 1))

        return _merge_segments(parent_segments)

