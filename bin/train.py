#!/usr/bin/env python3
"""
Train a U-Net segmentation model for datacenter detection.

This is a simpler alternative to the Clay Foundation Model that trains
all weights from scratch.

Usage:
    uv run python train_unet_segmentation.py \\
        --chips-dir data/chips \\
        --masks-dir data/masks \\
        --batch-size 8 \\
        --max-epochs 50
"""

import argparse
import sys
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins.precision import MixedPrecision

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datacenter_detector.data.datamodule import DatacenterSegmentationDataModule
from src.datacenter_detector.models.unet import UNetSegmentor


def main():
    parser = argparse.ArgumentParser(
        description="Train U-Net segmentation model for datacenter detection"
    )

    # Data arguments
    parser.add_argument(
        "--chips-dir",
        type=str,
        default="data/chips",
        help="Directory containing NAIP chip GeoTIFF files",
    )
    parser.add_argument(
        "--masks-dir",
        type=str,
        default="data/masks",
        help="Directory containing segmentation mask GeoTIFF files",
    )

    # Model arguments
    parser.add_argument(
        "--base-channels",
        type=int,
        default=32,
        help="Base number of channels in U-Net (32=light, 64=standard)",
    )
    parser.add_argument(
        "--class-weights",
        type=float,
        nargs=2,
        default=[1.0, 1.0],
        help="Class weights for loss [background, datacenter] (default: 1.0 1.0)",
    )

    # Training arguments
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--max-epochs", type=int, default=50, help="Maximum number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay")

    # Data split arguments
    parser.add_argument(
        "--train-ratio", type=float, default=0.7, help="Fraction of data for training"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15, help="Fraction of data for validation"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/unet",
        help="Directory for saving checkpoints and logs",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="unet_segmentation",
        help="Name for this experiment",
    )

    # Hardware arguments
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Accelerator type (auto, cpu, gpu, mps)",
    )
    parser.add_argument(
        "--devices", type=int, default=1, help="Number of devices to use"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("U-NET SEGMENTATION TRAINING")
    print("=" * 80)
    print(f"Chips directory: {args.chips_dir}")
    print(f"Masks directory: {args.masks_dir}")
    print(f"Base channels: {args.base_channels}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Class weights: {args.class_weights}")
    print(f"Accelerator: {args.accelerator}")

    # Check for hardware
    if torch.backends.mps.is_available():
        print("✓ Apple Silicon (MPS) detected and available")
    elif torch.cuda.is_available():
        print(f"✓ CUDA detected: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ Using CPU (training will be slower)")

    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    print()

    # Set seed for reproducibility
    L.seed_everything(args.seed)

    # Initialize data module
    print("Initializing data module...")
    datamodule = DatacenterSegmentationDataModule(
        chips_dir=args.chips_dir,
        masks_dir=args.masks_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # Initialize model
    print("Initializing U-Net model...")
    model = UNetSegmentor(
        lr=args.lr,
        wd=args.wd,
        class_weights=args.class_weights,
        base_channels=args.base_channels,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename=f"{args.experiment_name}-{{epoch:02d}}-{{val/iou:.4f}}",
        monitor="val/iou",
        mode="max",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )

    early_stopping = EarlyStopping(
        monitor="val/iou", patience=10, mode="max", verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Logger
    logger = TensorBoardLogger(save_dir=output_dir / "logs", name=args.experiment_name)

    # Trainer
    print("Initializing trainer...")

    # Determine accelerator and precision
    precision_plugin = None
    if args.accelerator == "auto":
        if torch.backends.mps.is_available():
            accelerator = "mps"
            precision_plugin = MixedPrecision(device="mps", precision="16-mixed")
        elif torch.cuda.is_available():
            accelerator = "gpu"
            precision_plugin = MixedPrecision(device="cuda", precision="16-mixed")
        else:
            accelerator = "cpu"
            precision_plugin = None
    else:
        accelerator = args.accelerator
        if accelerator in ["gpu", "cuda"]:
            precision_plugin = MixedPrecision(device="cuda", precision="16-mixed")
        elif accelerator == "mps":
            precision_plugin = MixedPrecision(device="mps", precision="16-mixed")
        else:
            precision_plugin = None

    print(
        f"Using accelerator: {accelerator}, precision: {'16-mixed' if precision_plugin else '32'}"
    )

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=args.devices,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        deterministic=True,
        plugins=[precision_plugin] if precision_plugin else None,
    )

    # Train
    print("\nStarting training...")
    print("=" * 80)
    trainer.fit(model, datamodule=datamodule)

    # Test on best model
    print("\n" + "=" * 80)
    print("Testing best model...")
    print("=" * 80)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best IoU score: {checkpoint_callback.best_model_score:.4f}")
    print(f"Logs saved to: {logger.log_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
