#!/usr/bin/env python3
"""
Download NAIP imagery as a mosaic from multiple tiles.
This ensures continuous coverage even when coordinates are near tile boundaries.

Usage:
    uv run python download_naip_mosaic.py \
        --lon -82.75066 \
        --lat 40.06605 \
        --size 2560 \
        --output test_images/datacenter_mosaic.tif
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import planetary_computer as pc
import rasterio
from pyproj import Transformer
from pystac_client import Client
from rasterio.merge import merge
from shapely.geometry import box


def download_naip_mosaic(lon, lat, size_pixels=2560, output_path="naip_mosaic.tif"):
    """
    Download NAIP imagery by creating a mosaic from all overlapping tiles.
    This ensures continuous coverage across tile boundaries.
    """
    NAIP_RESOLUTION_M = 0.6
    size_m = size_pixels * NAIP_RESOLUTION_M

    # Calculate the bounding box for the desired output area
    # Convert size to degrees (approximate)
    deg_per_m = 1.0 / 111000.0  # Approximate meters per degree at mid-latitudes
    half_size_deg = (size_m / 2) * deg_per_m

    # Output bbox centered on the point
    output_bbox = [
        lon - half_size_deg,
        lat - half_size_deg,
        lon + half_size_deg,
        lat + half_size_deg,
    ]

    # Search bbox - slightly larger to ensure we get all overlapping tiles
    search_margin = half_size_deg * 1.2  # 20% margin
    search_bbox = [
        lon - search_margin,
        lat - search_margin,
        lon + search_margin,
        lat + search_margin,
    ]

    print("=" * 80)
    print("NAIP MOSAIC DOWNLOAD (MULTI-TILE SUPPORT)")
    print("=" * 80)
    print(f"Center: ({lon:.6f}, {lat:.6f})")
    print(f"Size: {size_pixels}x{size_pixels} pixels (~{size_m:.0f}m × {size_m:.0f}m)")
    print(f"Output bbox: {[f'{x:.6f}' for x in output_bbox]}")
    print()

    # Search for NAIP
    print("Searching for NAIP imagery...")
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )

    search = catalog.search(
        collections=["naip"],
        bbox=search_bbox,
        datetime=f"2020-01-01/{datetime.now().strftime('%Y-%m-%d')}",
    )

    items = list(search.items())
    if not items:
        print("ERROR: No NAIP imagery found")
        return None

    print(f"✓ Found {len(items)} tiles in search area")

    # First pass: find ALL tiles that contain or are near the center point
    # Group them by date
    tiles_by_date = {}

    print("\nAnalyzing tiles:")
    for item in items:
        with rasterio.open(item.assets["image"].href) as src:
            # Transform center point to tile's CRS
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            center_x, center_y = transformer.transform(lon, lat)

            # Check if point is within or very near this tile
            buffer = size_m * 1.5  # 1.5x our output size

            tile_box = box(
                src.bounds.left - buffer,
                src.bounds.bottom - buffer,
                src.bounds.right + buffer,
                src.bounds.top + buffer,
            )

            point_box = box(center_x - 1, center_y - 1, center_x + 1, center_y + 1)

            if tile_box.intersects(point_box):
                contains = (
                    src.bounds.left <= center_x <= src.bounds.right
                    and src.bounds.bottom <= center_y <= src.bounds.top
                )
                date = item.datetime.date()

                if date not in tiles_by_date:
                    tiles_by_date[date] = []
                tiles_by_date[date].append((item, contains))

                status = "CONTAINS CENTER" if contains else "nearby"
                print(f"  {item.id} ({item.datetime.date()}) - {status}")

    if not tiles_by_date:
        print("\nERROR: No tiles found near the requested coordinates!")
        print(f"Center point: ({lon}, {lat})")
        print(f"Requested area: ~{size_m:.0f}m × {size_m:.0f}m")
        return None

    # Strategy: Prefer tiles that CONTAIN the center point, then most recent
    # Find the most recent date that has tiles containing the center
    dates_with_containing = []
    dates_with_nearby = []

    for date, tiles_and_status in tiles_by_date.items():
        has_containing = any(contains for item, contains in tiles_and_status)
        if has_containing:
            dates_with_containing.append(date)
        else:
            dates_with_nearby.append(date)

    if dates_with_containing:
        # Use most recent date that has tiles containing the center
        selected_date = max(dates_with_containing)
        tiles_and_status = tiles_by_date[selected_date]
        # Get all tiles from this date (containing + nearby for seamless mosaic)
        tiles_to_use = [item for item, contains in tiles_and_status]
        containing_count = sum(1 for item, contains in tiles_and_status if contains)
        print(f"\n✓ Using {len(tiles_to_use)} tile(s) from {selected_date}")
        print(
            f"  ({containing_count} contain center, {len(tiles_to_use) - containing_count} nearby for seamless mosaic)"
        )
    else:
        # Fall back to most recent nearby tiles
        selected_date = max(dates_with_nearby)
        tiles_and_status = tiles_by_date[selected_date]
        tiles_to_use = [item for item, contains in tiles_and_status]
        print(f"\n✓ Using {len(tiles_to_use)} nearby tile(s) from {selected_date}")
        print(
            "  WARNING: No tiles contain the exact center - output may not be centered"
        )

    for tile in tiles_to_use:
        print(f"  - {tile.id}")
    print()

    # Open all tiles
    print("Opening tiles and creating mosaic...")
    src_files = []
    for i, tile in enumerate(tiles_to_use, 1):
        print(f"  Opening tile {i}/{len(tiles_to_use)}: {tile.id}")
        src = rasterio.open(tile.assets["image"].href)
        src_files.append(src)

    # Merge tiles into a mosaic
    print(f"  Merging {len(src_files)} tiles (this may take a minute)...")
    mosaic, mosaic_transform = merge(src_files)
    print("  Merge complete!")

    # Get CRS from first tile (all should be the same)
    mosaic_crs = src_files[0].crs

    print(f"✓ Mosaic created: {mosaic.shape}")
    print(f"  CRS: {mosaic_crs}")
    print(f"  Transform: {mosaic_transform}")

    # Now extract the centered window from the mosaic
    # Transform center point to mosaic CRS
    transformer = Transformer.from_crs("EPSG:4326", mosaic_crs, always_xy=True)
    center_x, center_y = transformer.transform(lon, lat)

    print(f"\n  Center in projected coords: ({center_x:.2f}, {center_y:.2f})")

    # Convert to pixel coordinates in mosaic
    inv_transform = ~mosaic_transform
    center_col, center_row = inv_transform * (center_x, center_y)

    print(f"  Center pixel in mosaic: (col={center_col:.1f}, row={center_row:.1f})")

    # Calculate window centered on the point
    half_size = size_pixels // 2
    col_off = int(center_col - half_size)
    row_off = int(center_row - half_size)

    # Extract the window
    mosaic_height, mosaic_width = mosaic.shape[1], mosaic.shape[2]

    # Ensure window is within bounds
    if col_off < 0 or row_off < 0:
        print(
            f"  WARNING: Requested window starts outside mosaic (col={col_off}, row={row_off})"
        )
        col_off = max(0, col_off)
        row_off = max(0, row_off)

    if col_off + size_pixels > mosaic_width or row_off + size_pixels > mosaic_height:
        print(
            f"  WARNING: Requested window extends beyond mosaic ({mosaic_width}x{mosaic_height})"
        )

    # Clamp to valid range
    col_off = max(0, min(col_off, mosaic_width - size_pixels))
    row_off = max(0, min(row_off, mosaic_height - size_pixels))

    print(
        f"  Extracting window: col={col_off}, row={row_off}, size={size_pixels}x{size_pixels}"
    )

    # Extract window from mosaic
    output_data = mosaic[
        :, row_off : row_off + size_pixels, col_off : col_off + size_pixels
    ]

    # Calculate transform for the output window
    output_transform = mosaic_transform * rasterio.Affine.translation(col_off, row_off)

    # Close all source files
    for src in src_files:
        src.close()

    # Prepare output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write output
    profile = {
        "driver": "GTiff",
        "height": size_pixels,
        "width": size_pixels,
        "count": 4,
        "dtype": rasterio.uint8,
        "crs": mosaic_crs,
        "transform": output_transform,
        "compress": "lzw",
    }

    print(f"\nSaving to: {output_path}")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(output_data)

    file_size_mb = output_path.stat().st_size / 1024 / 1024

    print()
    print("=" * 80)
    print("SUCCESS")
    print("=" * 80)
    print(f"✓ Saved: {output_path}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Dimensions: {size_pixels}x{size_pixels} pixels")
    print(f"  Mosaicked from {len(tiles_to_use)} tile(s)")
    print(f"  Centered on ({lon:.6f}, {lat:.6f})")
    print("=" * 80)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Download NAIP mosaic from multiple tiles"
    )
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--size", type=int, default=2560)
    parser.add_argument("--output", type=str, default="naip_mosaic.tif")

    args = parser.parse_args()

    result = download_naip_mosaic(args.lon, args.lat, args.size, args.output)

    if not result:
        exit(1)


if __name__ == "__main__":
    main()
