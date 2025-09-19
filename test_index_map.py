#!/usr/bin/env python3
"""Demonstration of IndexMap with intermediate prints.

Pipeline of stages:
    orig  <- nogaps <- filtered <- final

We record a segment at 'final' and project it back to 'orig',
printing all intermediate steps and merges along the way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

# ---------------------------
# Library code (from your module)
# ---------------------------

Segment = Tuple[int, int]  # half-open [start, end)


def _merge_segments(segments: Iterable[Segment]) -> List[Segment]:
    """Merge overlapping or adjacent segments."""
    sorted_segments = sorted((s, e) for s, e in segments if e > s)
    if not sorted_segments:
        return []
    merged: List[Segment] = []
    cur_s, cur_e = sorted_segments[0]
    for s, e in sorted_segments[1:]:
        if s <= cur_e:  # overlap or adjacency (since half-open)
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


@dataclass(frozen=True)
class StageMapping:
    """Mapping metadata between two stages."""
    child: str
    parent: str
    kept_indices: Sequence[int]  # kept_indices[i_child] -> i_parent


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
        _stage, segs = self._segments[label]
        return list(segs)

    def project(self, label: str, target_stage: str) -> List[Segment]:
        """Project stored segments for *label* to the *target_stage* coordinate space."""
        if label not in self._segments:
            raise KeyError(f"Unknown label: {label}")

        stage, segments = self._segments[label]
        if stage == target_stage:
            return list(segments)

        projected: List[Segment] = list(segments)
        current_stage = stage
        chain: List[str] = [current_stage]

        while current_stage != target_stage:
            mapping = self._mappings.get(current_stage)
            if mapping is None:
                raise ValueError(
                    f"Cannot project from {stage} to {target_stage}: missing mapping for {current_stage}."
                )
            projected = self._project_segments_once(projected, mapping.kept_indices)
            current_stage = mapping.parent
            chain.append(current_stage)

        # Final cleanup
        return _merge_segments(projected)

    @staticmethod
    def _project_segments_once(segments: Sequence[Segment], kept_indices: Sequence[int]) -> List[Segment]:
        """Project segments from child stage to its parent using *kept_indices*."""
        if not segments or not kept_indices:
            return []

        parent_segments: List[Segment] = []
        N = len(kept_indices)
        for start, end in segments:
            if start >= end:
                continue
            # Clamp to child bounds
            s = max(start, 0)
            e = min(end, N)
            if s >= e:
                continue

            # Map each child index to its parent index
            parent_idxs = [kept_indices[i] for i in range(s, e)]
            if not parent_idxs:
                continue

            # Group consecutive parent indices into segments
            run_start = parent_idxs[0]
            prev = parent_idxs[0]
            for idx in parent_idxs[1:]:
                if idx != prev + 1:
                    parent_segments.append((run_start, prev + 1))
                    run_start = idx
                prev = idx
            parent_segments.append((run_start, prev + 1))

        return _merge_segments(parent_segments)

# ---------------------------
# Pretty-print helpers
# ---------------------------

def fmt_segments(segs: Sequence[Segment]) -> str:
    """Human-friendly segments formatter."""
    if not segs:
        return "∅"
    return " ".join(f"[{s},{e})" for s, e in segs)

def print_mapping(name_child: str, name_parent: str, kept: Sequence[int]) -> None:
    print(f"Mapping {name_child} -> {name_parent}:")
    print(f"  child indices:  0..{len(kept)-1}")
    print(f"  kept_indices:   {list(kept)} (values are indices in '{name_parent}')")
    print()

def explain_one_step(child_name: str, parent_name: str,
                     child_segments: Sequence[Segment],
                     kept_indices: Sequence[int]) -> List[Segment]:
    """Show how segments move one step up via kept_indices."""
    print(f"Projecting one step: {child_name} → {parent_name}")
    print(f"  input segments ({child_name}): {fmt_segments(child_segments)}")
    parent_segs = IndexMap._project_segments_once(child_segments, kept_indices)
    print(f"  output segments ({parent_name}): {fmt_segments(parent_segs)}\n")
    return parent_segs

# ---------------------------
# Example usage
# ---------------------------

if __name__ == "__main__":
    imap = IndexMap()

    # Define a simple chain: orig <- nogaps <- filtered <- final
    # Pretend orig has length 12; removing gaps produced 'nogaps' of length 9, etc.
    kept_nogaps_to_orig   = [0, 1, 2, 5, 6, 7, 9, 10, 11]     # len=9
    kept_filtered_to_nog  = [0, 1, 2, 4, 5, 6, 7]              # drop nogaps[3]; len=7
    kept_final_to_filt    = [0, 2, 3, 4, 6]                    # drop filtered[1] and [5]; len=5

    # Register mappings (child -> parent)
    imap.add_mapping("nogaps",   "orig",     kept_nogaps_to_orig)
    imap.add_mapping("filtered", "nogaps",   kept_filtered_to_nog)
    imap.add_mapping("final",    "filtered", kept_final_to_filt)

    # Show the mappings
    print_mapping("nogaps",   "orig",     kept_nogaps_to_orig)
    print_mapping("filtered", "nogaps",   kept_filtered_to_nog)
    print_mapping("final",    "filtered", kept_final_to_filt)

    # Record segments detected at 'final' (child-most stage)
    # Example: a detector marked samples [1,4) in 'final' as motion artifacts
    seg_final = [(1, 4)]
    imap.record_segments("motions", "final", seg_final)
    print(f"Recorded label 'motions' at stage 'final': {fmt_segments(seg_final)}\n")

    # Retrieve without projection
    stored = imap.get_segments("motions")
    print(f"Stored (no projection): {fmt_segments(stored)}\n")

    # --- Show intermediate projections step by step
    # final -> filtered
    seg_filtered = explain_one_step("final", "filtered", stored, kept_final_to_filt)

    # filtered -> nogaps
    seg_nogaps = explain_one_step("filtered", "nogaps", seg_filtered, kept_filtered_to_nog)

    # nogaps -> orig
    seg_orig_onehop = explain_one_step("nogaps", "orig", seg_nogaps, kept_nogaps_to_orig)

    # --- Now use the convenience multi-step projector and show the (merged) result
    seg_orig = imap.project("motions", target_stage="orig")
    print("Final projection via IndexMap.project(...):")
    print(f"  to 'orig': {fmt_segments(seg_orig)}")

    # Sanity check: the manual step-by-step and the full project() should match after merging
    assert seg_orig == _merge_segments(seg_orig_onehop), "Mismatch between manual and project() results!"

    # Extra: record another overlapping segment and show merged behavior
    imap.record_segments("motions2", "final", [(0, 2), (2, 3), (3, 5)])  # overlaps/adjacent -> will merge to [0,5)
    print("\nRecorded label 'motions2' at 'final':", fmt_segments(imap.get_segments("motions2")))
    print("Projected 'motions2' to 'orig':", fmt_segments(imap.project("motions2", "orig")))
