"""Utility for tracking index mappings between processing stages."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

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


def _is_segment_like(value: object) -> bool:
    """Best-effort check whether *value* looks like a (start, end) pair."""

    if isinstance(value, (tuple, list)):
        return len(value) == 2
    if hasattr(value, "__len__") and hasattr(value, "__getitem__"):
        try:
            return len(value) == 2
        except TypeError:
            return False
    return False


def _parse_kept(
    kept: Sequence[int] | Sequence[Segment] | Iterator[int] | Iterator[Segment],
    parent_length: Optional[int],
) -> Tuple[Tuple[Segment, ...], int]:
    """Normalize kept indices/runs into a tuple of parent runs and child length."""

    iterator: Iterator[object]
    if isinstance(kept, (list, tuple)):
        iterator = iter(kept)
    else:
        iterator = iter(kept)

    try:
        first = next(iterator)
    except StopIteration:
        return (), 0

    if _is_segment_like(first):
        runs: List[Segment] = []
        total = 0
        prev_end: Optional[int] = None
        for raw_segment in chain([first], iterator):
            start_raw, end_raw = raw_segment  # type: ignore[misc]
            start = int(start_raw)
            end = int(end_raw)
            if end <= start:
                raise ValueError("kept runs must have positive length")
            if start < 0:
                raise ValueError("kept runs must be non-negative")
            if parent_length is not None and end > parent_length:
                raise ValueError("kept runs exceed parent length")
            if prev_end is not None and start < prev_end:
                raise ValueError("kept runs must be in non-decreasing order and non-overlapping")
            runs.append((start, end))
            prev_end = end
            total += end - start
        return tuple(runs), total

    runs = []
    first_idx = int(first)  # type: ignore[arg-type]
    if first_idx < 0:
        raise ValueError("kept indices must be non-negative")
    if parent_length is not None and first_idx >= parent_length:
        raise ValueError("kept indices exceed parent length")
    run_start = first_idx
    prev_idx = first_idx
    total = 1

    for raw_idx in iterator:
        idx = int(raw_idx)  # type: ignore[arg-type]
        if idx < 0:
            raise ValueError("kept indices must be non-negative")
        if parent_length is not None and idx >= parent_length:
            raise ValueError("kept indices exceed parent length")
        if idx <= prev_idx:
            raise ValueError("kept indices must be strictly increasing")
        if idx != prev_idx + 1:
            runs.append((run_start, prev_idx + 1))
            run_start = idx
        prev_idx = idx
        total += 1

    runs.append((run_start, prev_idx + 1))
    return tuple(runs), total


@dataclass(frozen=True)
class StageMapping:
    """Mapping metadata between two stages."""

    child: str
    parent: str
    kept_runs: Tuple[Segment, ...]
    child_length: int
    parent_length: Optional[int] = None
    _prefix: Tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        prefix = [0]
        total = 0
        for start, end in self.kept_runs:
            length = end - start
            if length <= 0:
                raise ValueError("kept runs must have positive length")
            total += length
            prefix.append(total)
        if total != self.child_length:
            raise ValueError("child length must equal total kept run length")
        object.__setattr__(self, "_prefix", tuple(prefix))

    @property
    def prefix(self) -> Tuple[int, ...]:
        return self._prefix

    def is_empty(self) -> bool:
        return self.child_length == 0


class IndexMap:
    """Store removal metadata and project indices across stages."""

    def __init__(self) -> None:
        self._mappings: Dict[str, StageMapping] = {}
        self._segments: Dict[str, Tuple[str, List[Segment]]] = {}

    def add_mapping(
        self,
        child: str,
        parent: str,
        kept_indices: Sequence[int] | Sequence[Segment],
        *,
        parent_length: Optional[int] = None,
    ) -> None:
        """Record how indices from *child* map back to *parent*."""

        kept_runs, child_length = _parse_kept(kept_indices, parent_length)
        mapping = StageMapping(
            child=child,
            parent=parent,
            kept_runs=kept_runs,
            child_length=child_length,
            parent_length=parent_length,
        )
        self._mappings[child] = mapping

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
            projected = self._project_segments_once(projected, mapping)
            current_stage = mapping.parent

        return _merge_segments(projected)

    @staticmethod
    def _project_segments_once(
        segments: Sequence[Segment],
        kept: StageMapping | Sequence[int] | Sequence[Segment],
        *,
        parent_length: Optional[int] = None,
    ) -> List[Segment]:
        """Project segments from child stage to its parent."""

        if not segments:
            return []

        if isinstance(kept, StageMapping):
            mapping = kept
        else:
            kept_runs, child_length = _parse_kept(kept, parent_length)
            mapping = StageMapping(
                child="<temp>",
                parent="<temp>",
                kept_runs=kept_runs,
                child_length=child_length,
                parent_length=parent_length,
            )

        if mapping.is_empty():
            return []

        parent_segments: List[Segment] = []
        child_length = mapping.child_length
        prefix = mapping.prefix
        runs = mapping.kept_runs

        for start, end in segments:
            if start >= end:
                continue
            child_start = max(start, 0)
            child_end = min(end, child_length)
            if child_start >= child_end:
                continue

            cursor = child_start
            while cursor < child_end:
                run_idx = bisect_right(prefix, cursor) - 1
                run_start, _ = runs[run_idx]
                offset_in_run = cursor - prefix[run_idx]
                parent_run_start = run_start + offset_in_run
                run_capacity = prefix[run_idx + 1] - cursor
                span = min(run_capacity, child_end - cursor)
                parent_run_end = parent_run_start + span
                parent_segments.append((parent_run_start, parent_run_end))
                cursor += span

        return _merge_segments(parent_segments)
