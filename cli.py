"""Command line interface for the ECG denoising pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecg_pipeline import (
    ECGDenoisingPipeline,
    load_config,
    load_ecg_from_npy,
    load_segments_from_json,
    plot_ecg_with_annotations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ECG denoising pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/pipeline_config.json"),
        help="Path to the JSON configuration file.",
    )
    parser.add_argument(
        "--ecg-path",
        type=Path,
        default=Path("data/ecg_orig.npy"),
        help="Path to the `.npy` file containing the ECG signal.",
    )
    parser.add_argument(
        "--gaps-path",
        type=Path,
        default=Path("data/gaps_indexes.json"),
        help="Path to the JSON file containing gap segments.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where results will be written.",
    )
    parser.add_argument(
        "--save-arrays",
        action="store_true",
        help="Persist ecg_start and ecg_final arrays to disk.",
    )
    parser.add_argument(
        "--save-indexes",
        action="store_true",
        help="Persist detected segment metadata to JSON files.",
    )
    parser.add_argument(
        "--plot-file",
        type=Path,
        default=None,
        help="Optional path to save the annotated plot (PNG).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable on-screen plotting (useful for headless runs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    ecg_orig = load_ecg_from_npy(args.ecg_path)
    gaps_segments = load_segments_from_json(args.gaps_path)

    pipeline = ECGDenoisingPipeline(config)
    result = pipeline.run(ecg_orig, gaps_segments)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_arrays:
        np.save(output_dir / "ecg_start.npy", result.ecg_start)
        np.save(output_dir / "ecg_final.npy", result.ecg_final)

    if args.save_indexes:
        def dump_segments(name: str, segments):
            with (output_dir / f"{name}_segments.json").open("w", encoding="utf-8") as fh:
                json.dump([[int(s), int(e)] for s, e in segments], fh)

        dump_segments("gaps", result.gaps_segments)
        dump_segments("outliers", result.outlier_segments)
        dump_segments("rdropouts", result.rdropout_segments)
        dump_segments("outliers_projected", result.projected_outliers)
        dump_segments("rdropouts_projected", result.projected_rdropouts)

    plot_path = args.plot_file if args.plot_file is not None else (output_dir / "ecg_annotations.png")
    annotations = {
        "outliers": result.projected_outliers,
        "rdropouts": result.projected_rdropouts,
    }
    plot_ecg_with_annotations(
        result.ecg_start,
        sample_rate=config.plotting.sample_rate,
        annotations=annotations,
        total_points=config.plotting.total_points,
        title="ECG start with projected anomalies",
        output_path=plot_path,
        show=not args.no_show,
    )

    print("Pipeline completed.")
    print(f"Saved plot to: {plot_path}")
    if args.save_arrays:
        print(f"Saved arrays to: {output_dir}")
    if args.save_indexes:
        print("Segment metadata saved.")


if __name__ == "__main__":
    main()

