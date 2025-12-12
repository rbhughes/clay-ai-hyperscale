#!/usr/bin/env python3
"""
Run inference with trained U-Net model to detect datacenters and extract centroids.

This script:
1. Loads a trained U-Net checkpoint
2. Runs inference on NAIP chips
3. Extracts datacenter centroids with lat/lon coordinates
4. Handles datacenters spanning multiple chips via spatial merging

Usage:
    uv run python inference_unet.py \
        --checkpoint outputs/unet/checkpoints/best.ckpt \
        --chips-dir data/chips \
        --output-geojson predictions/datacenters.geojson \
        --merge-distance 50
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.features import shapes
from shapely.geometry import Point, Polygon, shape
from shapely.ops import unary_union
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datacenter_detector.data.dataset import SegmentationTransform
from src.datacenter_detector.models.unet import UNetSegmentor


def load_model(checkpoint_path, device):
    """Load trained U-Net model from checkpoint."""
    model = UNetSegmentor.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)
    return model


def predict_chip(model, chip_path, transform, device):
    """
    Run inference on a single chip.

    Returns:
        mask: Binary prediction mask (H, W)
        bounds: Chip bounds
        crs: Chip CRS
    """
    # Read chip
    with rasterio.open(chip_path) as src:
        pixels = src.read()  # (4, H, W)
        bounds = src.bounds
        crs = src.crs
        chip_transform = src.transform

    # Prepare input
    pixels = torch.from_numpy(pixels.astype(np.float32)) / 255.0

    # Apply normalization (no augmentation)
    if transform:
        pixels, _ = transform(pixels, torch.zeros(pixels.shape[1:]))

    pixels = pixels.unsqueeze(0).to(device)  # (1, 4, H, W)

    # Predict
    with torch.no_grad():
        logits = model(pixels)  # (1, 2, H, W)
        pred = torch.argmax(logits, dim=1).squeeze(0)  # (H, W)

    mask = pred.cpu().numpy().astype(np.uint8)

    return mask, bounds, crs, chip_transform


def extract_polygons_from_mask(mask, transform, crs, min_area_pixels=100):
    """
    Extract datacenter polygons from binary mask using connected components.

    Args:
        mask: Binary mask (H, W) with 0=background, 1=datacenter
        transform: Rasterio transform for georeferencing
        crs: Coordinate reference system
        min_area_pixels: Minimum polygon area in pixels

    Returns:
        GeoDataFrame with datacenter polygons
    """
    polygons = []

    # Extract shapes (polygons) from mask
    for geom, value in shapes(mask, transform=transform):
        if value == 1:  # Datacenter class
            poly = shape(geom)

            # Filter by minimum area
            if poly.area > min_area_pixels * (
                transform[0] ** 2
            ):  # Convert to map units
                polygons.append(
                    {
                        "geometry": poly,
                        "area_sqm": poly.area,
                    }
                )

    if not polygons:
        return gpd.GeoDataFrame(columns=["geometry", "area_sqm"], crs=crs)

    return gpd.GeoDataFrame(polygons, crs=crs)


def merge_overlapping_polygons(gdf, buffer_distance=10):
    """
    Merge polygons that overlap or are very close together.

    This handles datacenters split across chip boundaries.

    Args:
        gdf: GeoDataFrame with polygons
        buffer_distance: Distance in meters to buffer polygons before merging

    Returns:
        GeoDataFrame with merged polygons
    """
    if len(gdf) == 0:
        return gdf

    # Buffer polygons slightly to connect nearby pieces
    buffered = gdf.geometry.buffer(buffer_distance)

    # Merge overlapping buffered polygons
    merged = unary_union(buffered)

    # Un-buffer to get back to original size
    if hasattr(merged, "__iter__"):
        # Multiple separate polygons
        final_polys = [poly.buffer(-buffer_distance) for poly in merged.geoms]
    else:
        # Single polygon
        final_polys = [merged.buffer(-buffer_distance)]

    # Filter out any invalid/empty geometries
    final_polys = [
        p for p in final_polys if p.is_valid and not p.is_empty and p.area > 0
    ]

    # Create new GeoDataFrame
    merged_gdf = gpd.GeoDataFrame(
        {"geometry": final_polys, "area_sqm": [p.area for p in final_polys]},
        crs=gdf.crs,
    )

    return merged_gdf


def extract_centroids(polygons_gdf):
    """
    Extract centroids from polygons.

    Args:
        polygons_gdf: GeoDataFrame with datacenter polygons

    Returns:
        GeoDataFrame with centroid points (lat, lon)
    """
    if len(polygons_gdf) == 0:
        return gpd.GeoDataFrame(
            columns=["geometry", "lon", "lat", "area_sqm", "source_polygon_area"],
            crs=polygons_gdf.crs,
        )

    centroids = []

    for idx, row in polygons_gdf.iterrows():
        poly = row["geometry"]
        centroid = poly.centroid

        # Convert to lat/lon (EPSG:4326)
        centroid_latlon = (
            gpd.GeoDataFrame({"geometry": [centroid]}, crs=polygons_gdf.crs)
            .to_crs("EPSG:4326")
            .geometry[0]
        )

        centroids.append(
            {
                "geometry": centroid_latlon,
                "lon": centroid_latlon.x,
                "lat": centroid_latlon.y,
                "area_sqm": row["area_sqm"],
                "source_polygon_area": poly.area,
            }
        )

    return gpd.GeoDataFrame(centroids, crs="EPSG:4326")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with U-Net model to detect datacenters"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--chips-dir",
        type=str,
        default="data/chips",
        help="Directory containing NAIP chips",
    )
    parser.add_argument(
        "--output-geojson",
        type=str,
        default="predictions/datacenters_centroids.geojson",
        help="Output GeoJSON file for centroids",
    )
    parser.add_argument(
        "--output-polygons",
        type=str,
        default="predictions/datacenters_polygons.geojson",
        help="Output GeoJSON file for polygons (optional)",
    )
    parser.add_argument(
        "--merge-distance",
        type=float,
        default=50,
        help="Distance in meters to merge nearby detections (handles multi-chip datacenters)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=500,
        help="Minimum datacenter area in pixels to keep",
    )
    parser.add_argument(
        "--batch-process",
        action="store_true",
        help="Process all chips (default: process all)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.tif",
        help="Glob pattern for chip files",
    )

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    print("=" * 80)
    print("U-NET DATACENTER INFERENCE")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Chips directory: {args.chips_dir}")
    print(f"Device: {device}")
    print(f"Merge distance: {args.merge_distance}m")
    print(f"Min area: {args.min_area} pixels")
    print("=" * 80)
    print()

    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, device)
    print(f"✓ Model loaded from {args.checkpoint}")

    # Setup transform
    transform = SegmentationTransform(train=False)

    # Find chips
    chips_dir = Path(args.chips_dir)
    chip_files = sorted(chips_dir.glob(args.pattern))
    chip_files = [f for f in chip_files if "_mask" not in f.stem]
    print(f"Found {len(chip_files)} chips to process")
    print()

    # Process chips
    print("Running inference...")
    all_polygons = []

    for chip_path in tqdm(chip_files, desc="Processing chips"):
        try:
            # Predict
            mask, bounds, crs, chip_transform = predict_chip(
                model, chip_path, transform, device
            )

            # Skip if no detections
            if mask.sum() == 0:
                continue

            # Extract polygons
            polygons = extract_polygons_from_mask(
                mask, chip_transform, crs, min_area_pixels=args.min_area
            )

            if len(polygons) > 0:
                all_polygons.append(polygons)

        except Exception as e:
            print(f"\n⚠ Error processing {chip_path.name}: {e}")
            continue

    # Combine all polygons
    if not all_polygons:
        print("\n⚠ No datacenters detected!")
        return

    print(f"\n✓ Detected datacenters in {len(all_polygons)} chips")

    combined_polygons = gpd.GeoDataFrame(pd.concat(all_polygons, ignore_index=True))
    print(f"✓ Total polygon detections: {len(combined_polygons)}")

    # Merge overlapping polygons (handles multi-chip datacenters)
    print(f"\nMerging nearby polygons (distance: {args.merge_distance}m)...")
    merged_polygons = merge_overlapping_polygons(
        combined_polygons, buffer_distance=args.merge_distance
    )
    print(f"✓ After merging: {len(merged_polygons)} unique datacenters")

    # Extract centroids
    print("\nExtracting centroids...")
    centroids = extract_centroids(merged_polygons)
    print(f"✓ Extracted {len(centroids)} centroids")

    # Save results
    output_dir = Path(args.output_geojson).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    centroids.to_file(args.output_geojson, driver="GeoJSON")
    print(f"\n✓ Centroids saved to: {args.output_geojson}")

    if args.output_polygons:
        # Convert polygons to EPSG:4326 for GeoJSON
        merged_polygons_latlon = merged_polygons.to_crs("EPSG:4326")
        merged_polygons_latlon.to_file(args.output_polygons, driver="GeoJSON")
        print(f"✓ Polygons saved to: {args.output_polygons}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total datacenters found: {len(centroids)}")
    print(f"Average area: {centroids['area_sqm'].mean():.1f} m²")
    print(f"Min area: {centroids['area_sqm'].min():.1f} m²")
    print(f"Max area: {centroids['area_sqm'].max():.1f} m²")
    print("=" * 80)

    # Show first few centroids
    print("\nFirst 5 centroids:")
    print(centroids[["lat", "lon", "area_sqm"]].head())


if __name__ == "__main__":
    import pandas as pd

    main()
