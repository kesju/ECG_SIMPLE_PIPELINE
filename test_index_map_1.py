#!/usr/bin/env python3
"""
IndexMap demo: step-by-step projection UP (child→parent) and DOWN (parent→child).

Stages (linear chain):clear

    orig  <-  nogaps  <-  filtered  <-  final
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple, Optional

Segment = Tuple[int, int]  # half-open [start, end)

# ---------------------------
# Utilities
# ---------------------------

def merge_segments(segments: Iterable[Segment]) -> List[Segment]:
    """Merge overlapping or adjacent [start, end) segments."""
    sorted_segments = sorted((s, e) for s, e in segments if e > s)
    if not sorted_segments:
        return []
    out: List[Segment] = []
    cs, ce = sorted_segments[0]
    for s, e in sorted_segments[1:]:
        if s <= ce:  # overlap or adjacency
            ce = max(ce, e)
        else:
            out.append((cs, ce))
            cs, ce = s, e
    out.append((cs, ce))
    return out


def fmt_segments(segs: Sequence[Segment]) -> str:
    return " ".join(f"[{s},{e})" for s, e in segs) if segs else "∅"


# ---------------------------
# Core data structures
# ---------------------------

@dataclass(frozen=True)
class StageMapping:
    """Mapping from child stage to its parent stage."""
    child: str
    parent: str
    kept_indices: Sequence[int]  # kept_indices[i_child] -> i_parent


class IndexMap:
    """
    Stores mappings between linear stages and allows projecting segments:
      - UP:   child -> parent  (using kept_indices)
      - DOWN: parent -> child  (using inverse of kept_indices)
    """

    def __init__(self) -> None:
        # child -> mapping (child→parent)
        self._mappings: Dict[str, StageMapping] = {}
        # label -> (stage, segments)
        self._segments: Dict[str, Tuple[str, List[Segment]]] = {}

    # ----- Structure / topology -----

    def add_mapping(self, child: str, parent: str, kept_indices: Sequence[int]) -> None:
        """Register how child indices map into parent indices."""
        self._mappings[child] = StageMapping(child=child, parent=parent, kept_indices=tuple(kept_indices))

    def _parent_of(self, child: str) -> Optional[str]:
        m = self._mappings.get(child)
        return m.parent if m else None

    def _child_of(self, parent: str) -> Optional[str]:
        """
        For a linear chain, find the (single) child whose parent == parent.
        Returns None if not found.
        """
        for m in self._mappings.values():
            if m.parent == parent:
                return m.child
        return None

    # ----- Segment storage (optional convenience) -----

    def record_segments(self, label: str, stage: str, segments: Sequence[Segment]) -> None:
        self._segments[label] = (stage, merge_segments(segments))

    def get_segments(self, label: str) -> List[Segment]:
        stage, segs = self._segments[label]
        return list(segs)

    # ----- Projection (UP: child→parent) -----

    @staticmethod
    def _project_up_once(segments: Sequence[Segment], kept_indices: Sequence[int]) -> List[Segment]:
        """Project child segments to parent using kept_indices."""
        if not segments or not kept_indices:
            return []
        parent_segments: List[Segment] = []
        N = len(kept_indices)

        for s, e in segments:
            if s >= e:
                continue
            s = max(0, s)
            e = min(N, e)
            if s >= e:
                continue

            # Map every child index in [s,e) to parent index
            pidxs = [kept_indices[i] for i in range(s, e)]
            if not pidxs:
                continue

            # Compress consecutive parent indices into runs
            run_start = pidxs[0]
            prev = pidxs[0]
            for idx in pidxs[1:]:
                if idx != prev + 1:
                    parent_segments.append((run_start, prev + 1))
                    run_start = idx
                prev = idx
            parent_segments.append((run_start, prev + 1))

        return merge_segments(parent_segments)

    def project_up(self, start_stage: str, segments: Sequence[Segment], target_stage: str) -> List[Segment]:
        """Project segments from start_stage up to target_stage (following child→parent links)."""
        if start_stage == target_stage:
            return list(segments)

        cur_stage = start_stage
        cur_segments = list(segments)
        while cur_stage != target_stage:
            mapping = self._mappings.get(cur_stage)
            if mapping is None:
                raise ValueError(f"Missing mapping for stage '{cur_stage}' (cannot reach '{target_stage}').")
            cur_segments = self._project_up_once(cur_segments, mapping.kept_indices)
            cur_stage = mapping.parent
        return merge_segments(cur_segments)

    def project_label_up(self, label: str, target_stage: str) -> List[Segment]:
        """Project stored label segments upward to target_stage."""
        if label not in self._segments:
            raise KeyError(f"Unknown label: {label}")
        stage, segs = self._segments[label]
        return self.project_up(stage, segs, target_stage)

    # ----- Projection (DOWN: parent→child) -----

    @staticmethod
    def _project_down_once(parent_segments: Sequence[Segment], kept_indices: Sequence[int]) -> List[Segment]:
        """
        Project parent segments down to child using kept_indices.
        kept_indices maps child_i -> parent_index.
        For each child_i, if kept_indices[i] lies inside any parent segment, keep i.
        Then compress consecutive child indices into child segments.
        """
        if not parent_segments or not kept_indices:
            return []

        # Fast membership test for parent ranges: sort + sweep
        psegs = merge_segments(parent_segments)
        child_runs: List[Segment] = []

        # Scan child indices in ascending order and check whether their mapped parent index lies inside any range.
        def in_any_parent(idx: int) -> bool:
            # Binary search would be faster; a linear sweep is fine for a small demo.
            for s, e in psegs:
                if idx < s:
                    return False
                if s <= idx < e:
                    return True
            return False

        run_start: Optional[int] = None
        prev: Optional[int] = None

        for i_child, pidx in enumerate(kept_indices):
            if in_any_parent(pidx):
                if run_start is None:
                    run_start = i_child
                    prev = i_child
                elif i_child == (prev + 1):
                    prev = i_child
                else:
                    child_runs.append((run_start, prev + 1))
                    run_start, prev = i_child, i_child
            else:
                if run_start is not None:
                    child_runs.append((run_start, prev + 1))
                    run_start = prev = None

        if run_start is not None:
            child_runs.append((run_start, prev + 1))

        return merge_segments(child_runs)

    def project_down(self, start_stage: str, segments: Sequence[Segment], target_stage: str) -> List[Segment]:
        """Project segments from start_stage down to target_stage (following parent→child links)."""
        if start_stage == target_stage:
            return list(segments)

        cur_stage = start_stage
        cur_segments = list(segments)
        while cur_stage != target_stage:
            child = self._child_of(cur_stage)
            if child is None:
                raise ValueError(f"No child found for stage '{cur_stage}' (cannot reach '{target_stage}').")
            mapping = self._mappings[child]  # child→cur_stage(parent)
            cur_segments = self._project_down_once(cur_segments, mapping.kept_indices)
            cur_stage = child
        return merge_segments(cur_segments)


# ---------------------------
# Demo (step-by-step both directions)
# ---------------------------

def print_header(title: str) -> None:
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))

def print_mapping(child: str, parent: str, kept: Sequence[int]) -> None:
    print(f"Mapping {child} -> {parent}: kept_indices = {list(kept)}")

def main() -> None:
    im = IndexMap()

    # Linear chain: orig <- nogaps <- filtered <- final
    kept_nogaps_to_orig   = [0, 1, 2, 5, 6, 7, 9, 10, 11]  # len(child nogaps)=9
    kept_filtered_to_nog  = [0, 1, 2, 4, 5, 6, 7]          # len(child filtered)=7
    kept_final_to_filt    = [0, 2, 3, 4, 6]                # len(child final)=5

    im.add_mapping("nogaps",   "orig",     kept_nogaps_to_orig)
    im.add_mapping("filtered", "nogaps",   kept_filtered_to_nog)
    im.add_mapping("final",    "filtered", kept_final_to_filt)

    print_header("Stage mappings (child → parent)")
    print_mapping("nogaps",   "orig",     kept_nogaps_to_orig)
    print_mapping("filtered", "nogaps",   kept_filtered_to_nog)
    print_mapping("final",    "filtered", kept_final_to_filt)

    # ----------------- UPWARD WALK (final → orig)
    # Label detected at final
    label = "motions"
    seg_final = [(1, 4)]  # [1,4) in 'final'
    im.record_segments(label, "final", seg_final)

    print_header("UPWARD walk: final → filtered → nogaps → orig")
    print(f"final:    {fmt_segments(seg_final)}")

    step1 = im._project_up_once(seg_final, kept_final_to_filt)
    print(f"filtered: {fmt_segments(step1)}")

    step2 = im._project_up_once(step1, kept_filtered_to_nog)
    print(f"nogaps:   {fmt_segments(step2)}")

    step3 = im._project_up_once(step2, kept_nogaps_to_orig)
    print(f"orig:     {fmt_segments(step3)}")

    # Verify against multi-hop projector
    up_all = im.project_label_up(label, "orig")
    print(f"\nproject(label='{label}', target='orig') → {fmt_segments(up_all)}")

    # ----------------- DOWNWARD WALK (orig → final)
    # Suppose at 'orig' we take the projection we just got and push it DOWN
    seg_orig = up_all

    print_header("DOWNWARD walk: orig → nogaps → filtered → final")
    print(f"orig:     {fmt_segments(seg_orig)}")

    d1 = im._project_down_once(seg_orig, kept_nogaps_to_orig)
    print(f"nogaps:   {fmt_segments(d1)}")

    d2 = im._project_down_once(d1, kept_filtered_to_nog)
    print(f"filtered: {fmt_segments(d2)}")

    d3 = im._project_down_once(d2, kept_final_to_filt)
    print(f"final:    {fmt_segments(d3)}")

    # Sanity: after up then down we should contain at least the original 'final' region
    # (exact equality can differ if orig segments cover parent indices that never existed in deeper children)
    print_header("Sanity checks")
    print("Original final segments:      ", fmt_segments(seg_final))
    print("Up→Down recovered at 'final': ", fmt_segments(d3))

if __name__ == "__main__":
    main()
