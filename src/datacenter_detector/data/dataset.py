"""
PyTorch Dataset for datacenter semantic segmentation from NAIP imagery.
"""

from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class DatacenterSegmentationDataset(Dataset):
    """
    Dataset for loading NAIP chips and corresponding segmentation masks.

    Args:
        chips_dir: Directory containing NAIP chip GeoTIFF files
        masks_dir: Directory containing mask GeoTIFF files
        transform: Optional transform to apply to chips and masks
        split: One of 'train', 'val', or 'test'
        train_ratio: Fraction of data to use for training (default 0.7)
        val_ratio: Fraction of data to use for validation (default 0.15)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        chips_dir,
        masks_dir,
        transform=None,
        split="train",
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42,
    ):
        super().__init__()
        self.chips_dir = Path(chips_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.split = split

        # Find all chip files (exclude mask files)
        chip_files = sorted(self.chips_dir.glob("*.tif"))
        chip_files = [f for f in chip_files if "_mask" not in f.stem]

        # Filter to only chips that have corresponding masks and standard size (256x256)
        self.chip_files = []
        self.mask_files = []

        for chip_path in chip_files:
            mask_filename = chip_path.stem + "_mask.tif"
            mask_path = self.masks_dir / mask_filename

            if mask_path.exists():
                # Check chip size - only include 256x256 chips
                with rasterio.open(chip_path) as src:
                    if src.height == 256 and src.width == 256:
                        self.chip_files.append(chip_path)
                        self.mask_files.append(mask_path)

        if len(self.chip_files) == 0:
            raise ValueError(
                f"No matching chip-mask pairs found!\n"
                f"  Chips dir: {chips_dir}\n"
                f"  Masks dir: {masks_dir}"
            )

        # Split data into train/val/test
        np.random.seed(seed)
        indices = np.random.permutation(len(self.chip_files))

        n_train = int(len(indices) * train_ratio)
        n_val = int(len(indices) * val_ratio)

        if split == "train":
            indices = indices[:n_train]
        elif split == "val":
            indices = indices[n_train : n_train + n_val]
        elif split == "test":
            indices = indices[n_train + n_val :]
        else:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train', 'val', or 'test'"
            )

        self.chip_files = [self.chip_files[i] for i in indices]
        self.mask_files = [self.mask_files[i] for i in indices]

        print(f"{split.upper()} split: {len(self.chip_files)} chip-mask pairs")

    def __len__(self):
        return len(self.chip_files)

    def __getitem__(self, idx):
        """
        Load a chip and its corresponding mask.

        Returns:
            Dictionary containing:
                - pixels: (4, H, W) NAIP imagery (R, G, B, NIR) as float32
                - mask: (H, W) binary segmentation mask as int64
                - time: (4,) temporal coordinates
                - latlon: (4,) spatial coordinates
                - chip_path: Path to the chip file (for debugging)
        """
        chip_path = self.chip_files[idx]
        mask_path = self.mask_files[idx]

        # Read NAIP chip
        with rasterio.open(chip_path) as src:
            # Read all 4 bands (R, G, B, NIR)
            pixels = src.read()  # Shape: (4, H, W)

            # Get geospatial metadata
            bounds = src.bounds

            # Convert to center lat/lon
            center_lon = (bounds.left + bounds.right) / 2
            center_lat = (bounds.bottom + bounds.top) / 2

            # Get timestamp from metadata if available, otherwise use dummy
            try:
                from datetime import datetime

                # Try to parse timestamp from filename or metadata
                # For now, use a dummy timestamp (2020-01-01)
                timestamp = datetime(2020, 1, 1)
            except:
                from datetime import datetime

                timestamp = datetime(2020, 1, 1)

        # Read mask
        with rasterio.open(mask_path) as src:
            mask = src.read(1)  # Shape: (H, W)

        # Convert to torch tensors
        # Normalize pixels to [0, 1] range (NAIP is uint8 0-255)
        pixels = torch.from_numpy(pixels.astype(np.float32)) / 255.0
        mask = torch.from_numpy(mask.astype(np.int64))

        # Create temporal encoding (year, month, day, hour)
        time = torch.tensor(
            [
                timestamp.year,
                timestamp.month,
                timestamp.day,
                timestamp.hour,
            ],
            dtype=torch.float32,
        )

        # Create spatial encoding (center_lat, center_lon, center_lat, center_lon)
        # Clay expects 4D latlon encoding
        latlon = torch.tensor(
            [center_lat, center_lon, center_lat, center_lon],
            dtype=torch.float32,
        )

        # Apply transforms if provided
        if self.transform:
            # Note: Custom transforms should handle both pixels and mask together
            pixels, mask = self.transform(pixels, mask)

        return {
            "pixels": pixels,
            "mask": mask,
            "time": time,
            "latlon": latlon,
            "chip_path": str(chip_path),
        }


class SegmentationTransform:
    """
    Transform that applies the same geometric augmentation to both image and mask.
    """

    def __init__(self, train=True, mean=None, std=None):
        """
        Args:
            train: If True, apply data augmentation
            mean: Mean for normalization (4 values for RGBN)
            std: Std for normalization (4 values for RGBN)
        """
        self.train = train
        self.mean = mean if mean is not None else [0.485, 0.456, 0.406, 0.5]
        self.std = std if std is not None else [0.229, 0.224, 0.225, 0.25]

    def __call__(self, pixels, mask):
        """
        Apply transforms to pixels and mask.

        Args:
            pixels: (4, H, W) tensor
            mask: (H, W) tensor

        Returns:
            Transformed (pixels, mask) tuple
        """
        if self.train:
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                pixels = torch.flip(pixels, dims=[2])  # Flip width
                mask = torch.flip(mask, dims=[1])

            # Random vertical flip
            if torch.rand(1) > 0.5:
                pixels = torch.flip(pixels, dims=[1])  # Flip height
                mask = torch.flip(mask, dims=[0])

            # Random 90-degree rotation (0, 90, 180, or 270 degrees)
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                pixels = torch.rot90(pixels, k=k, dims=[1, 2])
                mask = torch.rot90(mask, k=k, dims=[0, 1])

        # Normalize pixels (already in [0, 1] range)
        mean = torch.tensor(self.mean).view(4, 1, 1)
        std = torch.tensor(self.std).view(4, 1, 1)
        pixels = (pixels - mean) / std

        return pixels, mask
