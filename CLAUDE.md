# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project for hyperscale Earth observation using the Clay Foundation Model. Clay is an open-source AI model built on masked autoencoder (MAE) architecture for processing satellite imagery from Sentinel-1 and Sentinel-2, generating embeddings for Earth observation applications.

The project uses the `claymodel` package (v1.5.0) from https://github.com/Clay-foundation/model as a git dependency.

## Environment Setup

This project uses `uv` as the package manager with Python 3.12+.

**Install dependencies:**
```bash
uv sync
```

**Activate virtual environment:**
```bash
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

**Run the main script:**
```bash
python main.py
# or with uv:
uv run python main.py
```

## Key Architecture

### Clay Foundation Model Components

The `claymodel` package provides these main components:

- **ClayMAEModule**: Lightning module wrapping the Clay MAE model with training logic
- **ClayDataModule**: Lightning data module for loading satellite imagery datasets
- **Model sizes**: `clay_mae_tiny`, `clay_mae_small`, `clay_mae_base`, `clay_mae_large`
- **Finetune modules**: Classification (`finetune.classify`), Segmentation (`finetune.segment`), Regression (`finetune.regression`), and Embedder tasks

### Model Checkpoint

The repository includes `clay-v1.5.ckpt` (5.2GB), which is the pre-trained Clay model checkpoint version 1.5. This checkpoint can be loaded into ClayMAEModule for inference or fine-tuning.

### Typical Usage Pattern

```python
from claymodel.module import ClayMAEModule
from claymodel.datamodule import ClayDataModule

# Load pre-trained model
model = ClayMAEModule.load_from_checkpoint("clay-v1.5.ckpt")

# Or create new model instance
model = ClayMAEModule(
    model_size="base",
    mask_ratio=0.75,
    patch_size=8,
    embeddings_level="mean"  # Options: "mean", "patch", "group"
)

# DataModule handles Sentinel-1/2 data loading
datamodule = ClayDataModule(...)
```

### Lightning CLI

The claymodel package uses PyTorch Lightning's LightningCLI for training workflows. Configuration is typically managed through YAML files and the metadata configuration is expected at `configs/metadata.yaml` (not present in this repo yet).

## Development Notes

- The project is in early stages with minimal custom code beyond the imported `claymodel` dependency
- The checkpoint file (`clay-v1.5.ckpt`) is tracked in git but ignored patterns suggest consideration for cloud storage in production
- No test suite exists yet
- PyTorch Lightning is the core training framework used by the underlying claymodel

## Package Management

- Uses `uv` for fast, modern Python package management
- Lock file (`uv.lock`) should be committed for reproducibility
- Clay model is installed from git source, not PyPI
- To update claymodel: modify `pyproject.toml` and run `uv sync`
