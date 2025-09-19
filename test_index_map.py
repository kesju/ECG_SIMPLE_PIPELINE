"""Unit-style checks for the IndexMap utilities."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ecg_pipeline.index_map import IndexMap, StageMapping

Segment = tuple[int, int]


def _expect_raises(exc: type[BaseException], func: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    try:
        func(*args, **kwargs)
    except exc as err:  # pragma: no cover - helper used in a few tests only
        return str(err)
    raise AssertionError(f"Expected {exc.__name__} to be raised")


def test_project_segments_once_with_dense_indices() -> None:
    segments: list[Segment] = [(1, 4)]
    kept = [0, 1, 2, 5, 6, 7]
    result = IndexMap._project_segments_once(segments, kept)
    assert result == [(1, 3), (5, 6)]


def test_project_segments_once_with_runs_descriptor() -> None:
    mapping = StageMapping(
        child="stage3",
        parent="stage2",
        kept_runs=((0, 4), (10, 12)),
        child_length=6,
        parent_length=20,
    )
    result = IndexMap._project_segments_once([(0, 6)], mapping)
    assert result == [(0, 4), (10, 12)]


def test_chain_projection_across_stages() -> None:
    imap = IndexMap()
    imap.record_segments("label", "stage3", [(1, 4)])

    imap.add_mapping("stage3", "stage2", [(0, 2), (3, 5)], parent_length=7)
    imap.add_mapping("stage2", "stage1", np.array([0, 1, 2, 4, 5, 6, 7]), parent_length=9)
    imap.add_mapping("stage1", "orig", [0, 1, 2, 5, 6, 7, 9, 10, 11], parent_length=12)

    projected = imap.project("label", "orig")
    assert projected == [(1, 3), (6, 8)]


def test_add_mapping_accepts_runs_and_computes_child_length() -> None:
    imap = IndexMap()
    imap.add_mapping("child", "parent", [(1, 3), (10, 12)], parent_length=20)
    mapping = imap._mappings["child"]
    assert mapping.child_length == 4
    assert mapping.kept_runs == ((1, 3), (10, 12))


def test_add_mapping_rejects_non_monotonic_indices() -> None:
    imap = IndexMap()
    message = _expect_raises(
        ValueError,
        imap.add_mapping,
        "child",
        "parent",
        [0, 2, 2],
        parent_length=5,
    )
    assert "strictly increasing" in message.lower()


def test_add_mapping_rejects_out_of_bounds_indices() -> None:
    imap = IndexMap()
    message = _expect_raises(
        ValueError,
        imap.add_mapping,
        "child",
        "parent",
        [0, 5],
        parent_length=5,
    )
    assert "parent length" in message.lower()


def test_add_mapping_rejects_overlapping_runs() -> None:
    imap = IndexMap()
    message = _expect_raises(
        ValueError,
        imap.add_mapping,
        "child",
        "parent",
        [(0, 3), (2, 4)],
        parent_length=10,
    )
    assert "non-decreasing" in message.lower()


if __name__ == "__main__":  # pragma: no cover - convenience runner
    for name, obj in sorted(globals().items()):
        if name.startswith("test_") and callable(obj):
            obj()
    print("All IndexMap checks passed.")
