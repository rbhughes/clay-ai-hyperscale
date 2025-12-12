#!/usr/bin/env python3
"""
Generate semantic segmentation masks from GeoJSON polygons and NAIP chips.

This script:
1. Reads datacenter polygon geometries from GeoJSON
2. For each NAIP chip, rasterizes intersecting polygons to a binary mask
3. Saves masks as GeoTIFF files with matching geospatial metadata
4. Generates validation images showing chip + mask overlay

Usage:
    python create_segmentation_masks.py \\
        --chips-dir data/chips \\
        --geojson-path data/datacenters.geojson \\
        --output-dir data/masks \\
        --buffer-meters 7.5 \\
        --validation-dir data/validation
"""

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.plot import show
from shapely.geometry import box
from tqdm import tqdm


def create_mask_from_chip(chip_path, polygons_gdf, buffer_meters=7.5, output_dir=None):
    """
    Create a binary segmentation mask for a single chip.

    Args:
        chip_path: Path to NAIP chip GeoTIFF
        polygons_gdf: GeoDataFrame containing datacenter polygons
        buffer_meters: Buffer distance in meters to expand polygons
        output_dir: Directory to save mask (if None, saves alongside chip)

    Returns:
        Tuple of (mask_array, mask_path, num_polygons)
    """
    # Read chip metadata
    with rasterio.open(chip_path) as src:
        chip_transform = src.transform
        chip_crs = src.crs
        chip_shape = (src.height, src.width)
        chip_bounds = src.bounds
        chip_profile = src.profile.copy()

    # Create bounding box for chip
    chip_bbox = box(*chip_bounds)

    # Reproject polygons to chip CRS if needed
    if polygons_gdf.crs != chip_crs:
        polygons_gdf = polygons_gdf.to_crs(chip_crs)

    # Find polygons that intersect this chip
    intersecting = polygons_gdf[polygons_gdf.intersects(chip_bbox)]

    if len(intersecting) == 0:
        # No polygons intersect - create empty mask
        mask = np.zeros(chip_shape, dtype=np.uint8)
        num_polygons = 0
    else:
        # Buffer polygons
        if buffer_meters > 0:
            buffered_geoms = intersecting.geometry.buffer(buffer_meters)
        else:
            buffered_geoms = intersecting.geometry

        # Rasterize polygons to mask
        # Union all polygons so overlapping areas are still just 1
        shapes = [(geom, 1) for geom in buffered_geoms if geom is not None]

        mask = rasterize(
            shapes=shapes,
            out_shape=chip_shape,
            transform=chip_transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,  # Include pixels that touch polygon boundary
        )
        num_polygons = len(intersecting)

    # Determine output path
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_filename = Path(chip_path).stem + "_mask.tif"
        mask_path = output_dir / mask_filename
    else:
        # Save alongside chip
        mask_path = Path(chip_path).parent / (Path(chip_path).stem + "_mask.tif")

    # Update profile for mask (single band, uint8)
    chip_profile.update(
        {
            "count": 1,
            "dtype": "uint8",
            "compress": "lzw",
            "nodata": None,
        }
    )

    # Write mask
    with rasterio.open(mask_path, "w", **chip_profile) as dst:
        dst.write(mask, 1)

    return mask, mask_path, num_polygons


def create_validation_image(chip_path, mask_path, output_dir, polygons_gdf=None):
    """
    Create a validation image showing chip RGB + mask overlay.

    Args:
        chip_path: Path to NAIP chip
        mask_path: Path to generated mask
        output_dir: Directory to save validation image
        polygons_gdf: Optional GeoDataFrame to overlay original polygons
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read chip (RGB bands only)
    with rasterio.open(chip_path) as src:
        # NAIP bands: R=0, G=1, B=2, NIR=3
        # Read RGB for visualization
        r = src.read(1)
        g = src.read(2)
        b = src.read(3)

        # Normalize to 0-1 for display (NAIP is 0-255 uint8)
        rgb = np.dstack([r, g, b])
        rgb = rgb / 255.0
        rgb = np.clip(rgb, 0, 1)  # Ensure valid range

    # Read mask
    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Original RGB
    axes[0].imshow(rgb)
    axes[0].set_title("NAIP RGB", fontsize=14)
    axes[0].axis("off")

    # Plot 2: Mask
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Segmentation Mask", fontsize=14)
    axes[1].axis("off")

    # Plot 3: RGB with mask overlay
    axes[2].imshow(rgb)
    # Overlay mask in red with transparency
    mask_overlay = np.zeros((*mask.shape, 4))
    mask_overlay[mask == 1] = [1, 0, 0, 0.5]  # Red with 50% transparency
    axes[2].imshow(mask_overlay)
    axes[2].set_title("RGB + Mask Overlay", fontsize=14)
    axes[2].axis("off")

    plt.suptitle(f"Validation: {Path(chip_path).stem}", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # Save validation image
    val_filename = Path(chip_path).stem + "_validation.png"
    val_path = output_dir / val_filename
    plt.savefig(val_path, dpi=150, bbox_inches="tight")
    plt.close()

    return val_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate segmentation masks from GeoJSON polygons and NAIP chips"
    )

    parser.add_argument(
        "--chips-dir",
        type=str,
        default="data/chips",
        help="Directory containing NAIP chip GeoTIFF files",
    )
    parser.add_argument(
        "--geojson-path",
        type=str,
        default="data/datacenters.geojson",
        help="Path to datacenters.geojson file with polygon geometries",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/masks",
        help="Directory to save generated mask files",
    )
    parser.add_argument(
        "--buffer-meters",
        type=float,
        default=7.5,
        help="Buffer distance in meters to expand polygons (default: 7.5)",
    )
    parser.add_argument(
        "--validation-dir",
        type=str,
        default="data/validation",
        help="Directory to save validation images (default: data/validation)",
    )
    parser.add_argument(
        "--validation-sample",
        type=int,
        default=None,
        help="Generate validation images for only N random chips (default: all chips with masks)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.tif",
        help="Glob pattern for chip files (default: *.tif)",
    )

    args = parser.parse_args()

    chips_dir = Path(args.chips_dir)
    output_dir = Path(args.output_dir)
    validation_dir = Path(args.validation_dir)

    print("=" * 80)
    print("SEGMENTATION MASK GENERATION")
    print("=" * 80)
    print(f"Chips directory: {chips_dir}")
    print(f"GeoJSON path: {args.geojson_path}")
    print(f"Output directory: {output_dir}")
    print(f"Buffer distance: {args.buffer_meters} meters")
    print(f"Validation directory: {validation_dir}")
    print("=" * 80)
    print()

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    # Load GeoJSON polygons
    print("Loading GeoJSON polygons...")
    polygons_gdf = gpd.read_file(args.geojson_path)
    print(f"✓ Loaded {len(polygons_gdf)} polygons")
    print(f"  CRS: {polygons_gdf.crs}")
    print()

    # Find all chip files
    chip_files = sorted(chips_dir.glob(args.pattern))
    # Exclude mask files if they exist
    chip_files = [f for f in chip_files if "_mask" not in f.stem]
    print(f"Found {len(chip_files)} chip files")
    print()

    # Process each chip
    print("Generating masks...")
    stats = {
        "total_chips": len(chip_files),
        "chips_with_polygons": 0,
        "total_polygons": 0,
        "empty_masks": 0,
    }

    chips_with_masks = []

    for chip_path in tqdm(chip_files, desc="Processing chips"):
        try:
            mask, mask_path, num_polygons = create_mask_from_chip(
                chip_path=chip_path,
                polygons_gdf=polygons_gdf,
                buffer_meters=args.buffer_meters,
                output_dir=output_dir,
            )

            if num_polygons > 0:
                stats["chips_with_polygons"] += 1
                stats["total_polygons"] += num_polygons
                chips_with_masks.append((chip_path, mask_path))
            else:
                stats["empty_masks"] += 1

        except Exception as e:
            print(f"\n⚠ Error processing {chip_path.name}: {e}")
            continue

    print()
    print("=" * 80)
    print("MASK GENERATION SUMMARY")
    print("=" * 80)
    print(f"Total chips processed: {stats['total_chips']}")
    print(f"Chips with datacenter polygons: {stats['chips_with_polygons']}")
    print(f"Chips with no polygons (empty masks): {stats['empty_masks']}")
    print(f"Total polygon instances rasterized: {stats['total_polygons']}")
    print(f"Masks saved to: {output_dir}")
    print("=" * 80)
    print()

    # Generate validation images
    if chips_with_masks:
        print("Generating validation images...")

        # Sample if requested
        if args.validation_sample and args.validation_sample < len(chips_with_masks):
            import random

            random.seed(42)
            validation_chips = random.sample(chips_with_masks, args.validation_sample)
            print(
                f"Sampling {args.validation_sample} of {len(chips_with_masks)} chips with masks"
            )
        else:
            validation_chips = chips_with_masks
            print(f"Generating for all {len(chips_with_masks)} chips with masks")

        for chip_path, mask_path in tqdm(
            validation_chips, desc="Creating validation images"
        ):
            try:
                create_validation_image(
                    chip_path=chip_path,
                    mask_path=mask_path,
                    output_dir=validation_dir,
                    polygons_gdf=polygons_gdf,
                )
            except Exception as e:
                print(f"\n⚠ Error creating validation image for {chip_path.name}: {e}")
                continue

        print()
        print("=" * 80)
        print(f"✓ Validation images saved to: {validation_dir}")
        print("=" * 80)

    print()
    print("DONE!")
    print()


if __name__ == "__main__":
    main()
