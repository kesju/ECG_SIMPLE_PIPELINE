# ECG Simple Pipeline

This project implements a modular ECG denoising pipeline with gap removal, outlier suppression, and R-dropout handling. The pipeline is configured via `config/pipeline_config.json` and exposes both a command-line interface (`cli.py`) and a Jupyter notebook (`cli.ipynb`) for interactive exploration.

## Project Layout

- `ecg_pipeline/`: Python package containing the pipeline implementation.
- `config/`: Configuration files used by the pipeline.
- `data/`: Sample ECG waveform (`.npy`) and gap metadata (`.json`).
- `outputs/`: Default directory for CLI artefacts (created on demand).
- `cli.py`: Command-line driver capable of loading data, saving arrays, indexes, and plots.
- `cli.ipynb`: Notebook demonstration of pipeline usage.

## Running the CLI

```bash
python cli.py --ecg-path data/ecg_orig.npy --gaps-path data/gaps_indexes.json --save-arrays --save-indexes --no-show
```

Outputs (arrays, JSON metadata, and plots) are written to the `outputs/` directory by default. Use `--plot-file <path>` to choose a different location.

## Notebook Demo

Open `cli.ipynb` in Jupyter to run the pipeline and render plots inline. The notebook loads the sample `.npy` waveform and `gaps_indexes.json` automatically.

