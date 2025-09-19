"""Main ECG denoising pipeline orchestrator."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .config import PipelineConfig
from .detection import detect_outliers, detect_r_dropouts
from .index_map import IndexMap, Segment
from .steps import remove_segments


@dataclass
class PipelineResult:
    """Container for pipeline outputs."""

    ecg_orig: np.ndarray
    ecg_start: np.ndarray
    ecg_final: np.ndarray
    gaps_segments: List[Segment]
    outlier_segments: List[Segment]
    rdropout_segments: List[Segment]
    projected_outliers: List[Segment]
    projected_rdropouts: List[Segment]
    index_map: IndexMap


def load_ecg_from_npy(path: Path) -> np.ndarray:
    """Load ECG samples from a `.npy` file as float32."""

    data = np.load(path, mmap_mode="r")
    array = np.asarray(data, dtype=np.float32)
    if isinstance(data, np.memmap):
        del data
    return array


def load_segments_from_json(path: Path) -> List[Segment]:
    """Load gap or anomaly segments from a JSON file."""

    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    segments: List[Segment] = []
    for item in payload:
        if isinstance(item, dict):
            start = int(item["start"])
            end = int(item["end"])
        else:
            start, end = item
        if end > start:
            segments.append((int(start), int(end)))
    return segments


class ECGDenoisingPipeline:
    """Pipeline that performs staged ECG denoising with index tracking."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.index_map = IndexMap()

    def run(
        self,
        ecg_orig: np.ndarray,
        gaps_segments: Sequence[Segment],
    ) -> PipelineResult:
        """Execute the denoising pipeline given pre-loaded data."""

        gaps_segments = list(gaps_segments)
        print("gaps_segments:", gaps_segments)
        self.index_map.record_segments("gaps", "ecg_orig", gaps_segments)
        print("Recorded gaps:", self.index_map.get_segments("gaps"))

        # Remove gaps
        ecg_start, kept_after_gaps = remove_segments(ecg_orig, gaps_segments)
        self.index_map.add_mapping(
            "ecg_start",
            "ecg_orig",
            kept_after_gaps,
            parent_length=len(ecg_orig),
        )
        print("Kept after gaps:", kept_after_gaps, len(kept_after_gaps), "of", len(ecg_orig))

        # Detect & remove outliers
        cfg = self.config
        outlier_segments = detect_outliers(
            ecg_start,
            amplitude_threshold=cfg.outlier_detection.amplitude_threshold,
            min_segment_length=cfg.outlier_detection.min_segment_length,
        )
        print("outlier_segments:", outlier_segments)
        self.index_map.record_segments("outliers", "ecg_start", outlier_segments)

        ecg_no_outliers, kept_after_outliers = remove_segments(ecg_start, outlier_segments)
        self.index_map.add_mapping(
            "ecg_no_outliers",
            "ecg_start",
            kept_after_outliers,
            parent_length=len(ecg_start),
        )

        # Detect & remove R-dropouts
        rdropout_segments = detect_r_dropouts(
            ecg_no_outliers,
            std_threshold=cfg.rdropout_detection.std_threshold,
            window_size=cfg.rdropout_detection.window_size,
            min_segment_length=cfg.rdropout_detection.min_segment_length,
        )
        self.index_map.record_segments("rdropouts", "ecg_no_outliers", rdropout_segments)

        ecg_final, kept_after_rdropouts = remove_segments(ecg_no_outliers, rdropout_segments)
        self.index_map.add_mapping(
            "ecg_final",
            "ecg_no_outliers",
            kept_after_rdropouts,
            parent_length=len(ecg_no_outliers),
        )

        # Memory optimization: ecg_no_outliers no longer needed.
        del ecg_no_outliers

        projected_outliers = self.index_map.project("outliers", "ecg_start")
        projected_rdropouts = self.index_map.project("rdropouts", "ecg_start")

        return PipelineResult(
            ecg_orig=ecg_orig,
            ecg_start=ecg_start,
            ecg_final=ecg_final,
            gaps_segments=list(gaps_segments),
            outlier_segments=outlier_segments,
            rdropout_segments=rdropout_segments,
            projected_outliers=projected_outliers,
            projected_rdropouts=projected_rdropouts,
            index_map=self.index_map,
        )
