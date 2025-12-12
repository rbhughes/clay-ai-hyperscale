"""
PyTorch Lightning DataModule for datacenter semantic segmentation.
"""

import lightning as L
import torch
from torch.utils.data import DataLoader

from .dataset import (
    DatacenterSegmentationDataset,
    SegmentationTransform,
)


class DatacenterSegmentationDataModule(L.LightningDataModule):
    """
    DataModule for loading NAIP chips and segmentation masks.

    Args:
        chips_dir: Directory containing NAIP chip GeoTIFF files
        masks_dir: Directory containing mask GeoTIFF files
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        train_ratio: Fraction of data to use for training (default 0.7)
        val_ratio: Fraction of data to use for validation (default 0.15)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        chips_dir,
        masks_dir,
        batch_size=8,
        num_workers=4,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42,
    ):
        super().__init__()
        self.chips_dir = chips_dir
        self.masks_dir = masks_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed

        # NAIP imagery statistics (approximate)
        self.mean = [0.485, 0.456, 0.406, 0.5]  # R, G, B, NIR
        self.std = [0.229, 0.224, 0.225, 0.25]  # R, G, B, NIR

        # Create transforms
        self.trn_tfm = SegmentationTransform(train=True, mean=self.mean, std=self.std)
        self.val_tfm = SegmentationTransform(train=False, mean=self.mean, std=self.std)

    def setup(self, stage=None):
        """
        Setup datasets for each stage.

        Args:
            stage: 'fit', 'validate', 'test', or None
        """
        if stage in {"fit", None}:
            self.trn_ds = DatacenterSegmentationDataset(
                chips_dir=self.chips_dir,
                masks_dir=self.masks_dir,
                transform=self.trn_tfm,
                split="train",
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                seed=self.seed,
            )

            self.val_ds = DatacenterSegmentationDataset(
                chips_dir=self.chips_dir,
                masks_dir=self.masks_dir,
                transform=self.val_tfm,
                split="val",
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                seed=self.seed,
            )

        if stage in {"test", None}:
            self.test_ds = DatacenterSegmentationDataset(
                chips_dir=self.chips_dir,
                masks_dir=self.masks_dir,
                transform=self.val_tfm,
                split="test",
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                seed=self.seed,
            )

    def train_dataloader(self):
        """Returns DataLoader for training dataset."""
        # Disable pin_memory on MPS (Apple Silicon) as it's not supported
        pin_mem = torch.cuda.is_available()

        return DataLoader(
            self.trn_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=pin_mem,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        """Returns DataLoader for validation dataset."""
        # Disable pin_memory on MPS (Apple Silicon) as it's not supported
        pin_mem = torch.cuda.is_available()

        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_mem,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        """Returns DataLoader for test dataset."""
        # Disable pin_memory on MPS (Apple Silicon) as it's not supported
        pin_mem = torch.cuda.is_available()

        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_mem,
            persistent_workers=True if self.num_workers > 0 else False,
        )
