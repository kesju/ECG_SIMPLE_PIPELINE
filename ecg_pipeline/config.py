"""Configuration utilities for the ECG denoising pipeline."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class OutlierDetectionConfig:
    """Parameters for the mock outlier detection routine."""

    amplitude_threshold: float
    min_segment_length: int


@dataclass(frozen=True)
class RDropoutDetectionConfig:
    """Parameters for the mock R-dropout detection routine."""

    std_threshold: float
    window_size: int
    min_segment_length: int


@dataclass(frozen=True)
class PlottingConfig:
    """Plotting-related parameters."""

    sample_rate: int
    total_points: int | None


@dataclass(frozen=True)
class PipelineConfig:
    """Aggregated pipeline configuration."""

    outlier_detection: OutlierDetectionConfig
    rdropout_detection: RDropoutDetectionConfig
    plotting: PlottingConfig


def load_config(config_path: Path) -> PipelineConfig:
    """Load pipeline configuration from a JSON file."""

    with config_path.open("r", encoding="utf-8") as fh:
        payload: Dict[str, Any] = json.load(fh)

    outlier_section = payload.get("outlier_detection", {})
    rdrop_section = payload.get("rdropout_detection", {})
    plotting_section = payload.get("plotting", {})

    config = PipelineConfig(
        outlier_detection=OutlierDetectionConfig(
            amplitude_threshold=float(outlier_section["amplitude_threshold"]),
            min_segment_length=int(outlier_section["min_segment_length"]),
        ),
        rdropout_detection=RDropoutDetectionConfig(
            std_threshold=float(rdrop_section["std_threshold"]),
            window_size=int(rdrop_section["window_size"]),
            min_segment_length=int(rdrop_section["min_segment_length"]),
        ),
        plotting=PlottingConfig(
            sample_rate=int(plotting_section["sample_rate"]),
            total_points=int(plotting_section["total_points"]) if plotting_section.get("total_points") is not None else None,
        ),
    )

    return config

