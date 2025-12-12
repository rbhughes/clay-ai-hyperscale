"""
Simple U-Net model for datacenter semantic segmentation.

This is a lightweight alternative to the Clay Foundation Model that trains
all weights from scratch, which may work better for specific tasks like
datacenter detection.
"""

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchmetrics import Metric
from torchmetrics.classification import BinaryJaccardIndex


class DoubleConv(nn.Module):
    """Double convolution block used in U-Net."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    Lightweight U-Net for binary segmentation.

    Args:
        in_channels: Number of input channels (4 for RGBN NAIP)
        num_classes: Number of output classes (2 for binary)
        base_channels: Number of channels in first layer (default: 32)
    """

    def __init__(self, in_channels=4, num_classes=2, base_channels=32):
        super().__init__()

        # Encoder (downsampling)
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)

        # Decoder (upsampling)
        self.upconv4 = nn.ConvTranspose2d(
            base_channels * 16, base_channels * 8, kernel_size=2, stride=2
        )
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)

        self.upconv3 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4, kernel_size=2, stride=2
        )
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)

        self.upconv2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.upconv1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        # Output
        self.out = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        return self.out(dec1)


class UNetSegmentor(L.LightningModule):
    """
    Lightning module for U-Net segmentation.

    Args:
        lr: Learning rate
        wd: Weight decay
        class_weights: Optional class weights for loss
        base_channels: Base number of channels in U-Net (smaller = faster)
    """

    def __init__(
        self,
        lr=1e-3,
        wd=1e-4,
        class_weights=None,
        base_channels=32,
    ):
        super().__init__()
        self.save_hyperparameters()

        # U-Net model
        self.model = UNet(in_channels=4, num_classes=2, base_channels=base_channels)

        # Loss function
        if class_weights is not None:
            weight = torch.tensor(class_weights, dtype=torch.float32)
        else:
            weight = torch.tensor([1.0, 1.0], dtype=torch.float32)

        self.loss_fn = nn.CrossEntropyLoss(weight=weight)

        # Metrics
        self.train_iou = BinaryJaccardIndex()
        self.train_dice = BinaryDiceScore()

        self.val_iou = BinaryJaccardIndex()
        self.val_dice = BinaryDiceScore()
        self.val_precision = BinaryPrecisionSegmentation()
        self.val_recall = BinaryRecallSegmentation()

    def forward(self, pixels):
        """Forward pass - just takes pixel tensor."""
        return self.model(pixels)

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/iou",
                "interval": "epoch",
            },
        }

    def training_step(self, batch, batch_idx):
        """Training step."""
        pixels = batch["pixels"]
        masks = batch["mask"]
        batch_size = masks.size(0)

        logits = self(pixels)
        loss = self.loss_fn(logits, masks)

        preds = torch.argmax(logits, dim=1)
        self.train_iou(preds, masks)
        self.train_dice(preds, masks)

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "train/iou",
            self.train_iou,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "train/dice",
            self.train_dice,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        pixels = batch["pixels"]
        masks = batch["mask"]
        batch_size = masks.size(0)

        logits = self(pixels)
        loss = self.loss_fn(logits, masks)

        preds = torch.argmax(logits, dim=1)
        self.val_iou(preds, masks)
        self.val_dice(preds, masks)
        self.val_precision(preds, masks)
        self.val_recall(preds, masks)

        self.log(
            "val/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "val/iou",
            self.val_iou,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "val/dice",
            self.val_dice,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "val/precision",
            self.val_precision,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "val/recall",
            self.val_recall,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        return self.validation_step(batch, batch_idx)


# Reuse the custom metrics from datacenter_segmentation_model
class BinaryPrecisionSegmentation(Metric):
    """Compute precision for binary segmentation."""

    def __init__(self):
        super().__init__()
        self.add_state("tp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds, target):
        self.tp += ((preds == 1) & (target == 1)).sum().float()
        self.fp += ((preds == 1) & (target == 0)).sum().float()

    def compute(self):
        if self.tp + self.fp == 0:
            return torch.tensor(0.0)
        return self.tp / (self.tp + self.fp)


class BinaryRecallSegmentation(Metric):
    """Compute recall for binary segmentation."""

    def __init__(self):
        super().__init__()
        self.add_state("tp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds, target):
        self.tp += ((preds == 1) & (target == 1)).sum().float()
        self.fn += ((preds == 0) & (target == 1)).sum().float()

    def compute(self):
        if self.tp + self.fn == 0:
            return torch.tensor(0.0)
        return self.tp / (self.tp + self.fn)


class BinaryDiceScore(Metric):
    """Compute Dice score for binary segmentation."""

    def __init__(self):
        super().__init__()
        self.add_state("tp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds, target):
        self.tp += ((preds == 1) & (target == 1)).sum().float()
        self.fp += ((preds == 1) & (target == 0)).sum().float()
        self.fn += ((preds == 0) & (target == 1)).sum().float()

    def compute(self):
        if 2 * self.tp + self.fp + self.fn == 0:
            return torch.tensor(0.0)
        return (2 * self.tp) / (2 * self.tp + self.fp + self.fn)
