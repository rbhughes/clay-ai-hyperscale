#!/usr/bin/env python3
"""
Run inference on a single NAIP image of any size.

This script:
1. Tiles large images into 256x256 chips
2. Runs U-Net inference on each chip
3. Stitches predictions back together
4. Extracts datacenter centroids

Usage:
    uv run python inference_single_image.py \
        --checkpoint outputs/unet/checkpoints/best.ckpt \
        --image path/to/naip_image.tif \
        --output-geojson predictions/centroids.geojson \
        --output-mask predictions/prediction_mask.tif
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datacenter_detector.data.dataset import SegmentationTransform
from src.datacenter_detector.models.unet import UNetSegmentor


def tile_image(image, tile_size=256, overlap=0):
    """
    Tile a large image into smaller chips.

    Args:
        image: (C, H, W) array
        tile_size: Size of each tile (default: 256)
        overlap: Overlap in pixels between tiles (default: 0)

    Returns:
        List of (tile, row_offset, col_offset)
    """
    C, H, W = image.shape
    tiles = []

    stride = tile_size - overlap

    for i in range(0, H, stride):
        for j in range(0, W, stride):
            # Extract tile
            tile = image[:, i : min(i + tile_size, H), j : min(j + tile_size, W)]

            # Pad if needed to reach tile_size
            if tile.shape[1] < tile_size or tile.shape[2] < tile_size:
                padded = np.zeros((C, tile_size, tile_size), dtype=tile.dtype)
                padded[:, : tile.shape[1], : tile.shape[2]] = tile
                tile = padded

            tiles.append((tile, i, j))

    return tiles


def predict_single_image(model, image_path, device, tile_size=256, overlap=32):
    """
    Run inference on a single image of any size.

    Args:
        model: Trained U-Net model
        image_path: Path to NAIP image
        device: Device to run on
        tile_size: Size of tiles for processing
        overlap: Overlap between tiles to avoid edge artifacts

    Returns:
        prediction_mask: Binary mask (H, W)
        transform: Rasterio transform
        crs: Coordinate reference system
    """
    # Read image
    with rasterio.open(image_path) as src:
        image = src.read()  # (4, H, W)
        image_transform = src.transform
        crs = src.crs
        H, W = image.shape[1:]

    # Normalize
    image = image.astype(np.float32) / 255.0

    # Tile image
    tiles = tile_image(image, tile_size=tile_size, overlap=overlap)

    # Prepare transform
    transform = SegmentationTransform(train=False)

    # Process each tile
    prediction_mask = np.zeros((H, W), dtype=np.uint8)
    count_mask = np.zeros((H, W), dtype=np.float32)  # For averaging overlaps

    print(f"Processing {len(tiles)} tiles...")

    skipped_tiles = 0

    for tile, row_offset, col_offset in tiles:
        # Skip tiles that are mostly nodata/zeros (padded regions or masked areas)
        # Check if tile has significant data (more than 10% non-zero pixels)
        non_zero_ratio = np.count_nonzero(tile) / tile.size
        if non_zero_ratio < 0.1:
            skipped_tiles += 1
            continue

        # Prepare input
        tile_tensor = torch.from_numpy(tile)
        tile_tensor, _ = transform(tile_tensor, torch.zeros(tile_tensor.shape[1:]))
        tile_tensor = tile_tensor.unsqueeze(0).to(device)  # (1, 4, 256, 256)

        # Predict
        with torch.no_grad():
            logits = model(tile_tensor)
            # Get probability for datacenter class
            probs = torch.softmax(logits, dim=1)[0, 1]  # (256, 256)

        probs = probs.cpu().numpy()

        # Determine actual tile size (may be padded)
        actual_h = min(tile_size, H - row_offset)
        actual_w = min(tile_size, W - col_offset)

        # Add to prediction (weighted average for overlaps)
        prediction_mask[
            row_offset : row_offset + actual_h, col_offset : col_offset + actual_w
        ] += (probs[:actual_h, :actual_w] > 0.5).astype(np.uint8)
        count_mask[
            row_offset : row_offset + actual_h, col_offset : col_offset + actual_w
        ] += 1

    if skipped_tiles > 0:
        print(f"  Skipped {skipped_tiles} tiles with nodata/padding")

    # Average overlapping regions (use majority vote)
    # Avoid division by zero for pixels that were never processed
    with np.errstate(divide="ignore", invalid="ignore"):
        prediction_mask = (prediction_mask / count_mask > 0.5).astype(np.uint8)

    # Set unprocessed pixels (count_mask == 0) to background
    prediction_mask[count_mask == 0] = 0

    return prediction_mask, image_transform, crs


def extract_centroids(mask, transform, crs, min_area_pixels=100, merge_distance_m=10):
    """
    Extract datacenter centroids from prediction mask.

    Args:
        mask: Binary prediction mask
        transform: Rasterio transform
        crs: Coordinate reference system
        min_area_pixels: Minimum area in pixels for a polygon
        merge_distance_m: Distance in meters to merge nearby polygons (default: 10m)
    """
    polygons = []

    # Extract shapes
    for geom, value in shapes(mask, transform=transform):
        if value == 1:
            poly = shape(geom)

            # Filter by area
            area_sqm = poly.area
            area_pixels = area_sqm / (transform[0] ** 2)

            if area_pixels >= min_area_pixels:
                polygons.append(poly)

    if not polygons:
        return gpd.GeoDataFrame(columns=["geometry", "lat", "lon", "area_sqm"], crs=crs)

    print(f"  Found {len(polygons)} raw polygons")

    # Filter edge polygons BEFORE merging to prevent edge artifacts from being merged with real detections
    H, W = mask.shape
    edge_buffer_pixels = 50
    polygons_filtered = []
    edge_filtered_pre = 0

    for poly in polygons:
        bounds = poly.bounds
        inv_transform = ~transform
        min_col, max_row = inv_transform * (bounds[0], bounds[1])
        max_col, min_row = inv_transform * (bounds[2], bounds[3])

        touches_edge = (
            min_col < edge_buffer_pixels
            or min_row < edge_buffer_pixels
            or max_col > W - edge_buffer_pixels
            or max_row > H - edge_buffer_pixels
        )

        if touches_edge:
            edge_filtered_pre += 1
        else:
            polygons_filtered.append(poly)

    if edge_filtered_pre > 0:
        print(f"  Filtered {edge_filtered_pre} edge polygon(s) before merging")

    if not polygons_filtered:
        return gpd.GeoDataFrame(columns=["geometry", "lat", "lon", "area_sqm"], crs=crs)

    polygons = polygons_filtered

    # Merge nearby polygons to handle fragmentation from tiled inference
    # Buffer -> Union -> Negative buffer
    print(f"  Merging polygons within {merge_distance_m}m...")

    # Create GeoDataFrame for efficient operations
    gdf_polys = gpd.GeoDataFrame({"geometry": polygons}, crs=crs)

    # Buffer all polygons
    buffered = gdf_polys.buffer(merge_distance_m)

    # Union all overlapping buffered polygons
    merged_buffered = unary_union(buffered)

    # Negative buffer to return to original size (approximately)
    if hasattr(merged_buffered, "geoms"):
        # MultiPolygon
        merged = [geom.buffer(-merge_distance_m) for geom in merged_buffered.geoms]
    else:
        # Single Polygon
        merged = [merged_buffered.buffer(-merge_distance_m)]

    # Filter out any invalid or tiny geometries created by buffering
    # Also filter out polygons touching image edges (likely false positives from padding)
    H, W = mask.shape
    edge_buffer_pixels = 50  # Increased buffer to catch edge artifacts (50px = ~15m)
    merged_valid = []
    edge_filtered = 0

    for geom in merged:
        if geom.is_valid and not geom.is_empty and geom.area > 0:
            area_pixels = geom.area / (transform[0] ** 2)
            if area_pixels >= min_area_pixels:
                # Check if polygon touches image edge
                bounds = geom.bounds  # (minx, miny, maxx, maxy) in CRS coordinates

                # Convert bounds to pixel coordinates
                # bounds format: (minx, miny, maxx, maxy)
                # transform maps (x,y) in CRS coords -> (col, row) in pixel coords
                inv_transform = ~transform
                min_col, max_row = inv_transform * (bounds[0], bounds[1])  # bottom-left
                max_col, min_row = inv_transform * (bounds[2], bounds[3])  # top-right

                # Check if polygon is too close to image edges
                touches_edge = (
                    min_col < edge_buffer_pixels
                    or min_row < edge_buffer_pixels
                    or max_col > W - edge_buffer_pixels
                    or max_row > H - edge_buffer_pixels
                )

                if touches_edge:
                    edge_filtered += 1
                else:
                    merged_valid.append(geom)

    if edge_filtered > 0:
        print(f"  Filtered {edge_filtered} polygon(s) touching image edges")

    print(f"  After merging: {len(merged_valid)} datacenter(s)")

    if not merged_valid:
        return gpd.GeoDataFrame(columns=["geometry", "lat", "lon", "area_sqm"], crs=crs)

    # Extract centroids from merged polygons
    centroids = []
    for poly in merged_valid:
        centroid = poly.centroid
        area_sqm = poly.area
        area_pixels = area_sqm / (transform[0] ** 2)

        # Convert to lat/lon
        centroid_latlon = (
            gpd.GeoDataFrame({"geometry": [centroid]}, crs=crs)
            .to_crs("EPSG:4326")
            .geometry[0]
        )

        centroids.append(
            {
                "geometry": centroid_latlon,
                "lon": centroid_latlon.x,
                "lat": centroid_latlon.y,
                "area_sqm": area_sqm,
                "area_pixels": area_pixels,
            }
        )

    return gpd.GeoDataFrame(centroids, crs="EPSG:4326")


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single NAIP image")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained U-Net checkpoint",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to NAIP image (any size)",
    )
    parser.add_argument(
        "--output-geojson",
        type=str,
        default="predictions/centroids.geojson",
        help="Output GeoJSON file for centroids",
    )
    parser.add_argument(
        "--output-mask",
        type=str,
        default=None,
        help="Optional: Save prediction mask as GeoTIFF",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="Tile size for processing (default: 256)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=32,
        help="Overlap between tiles in pixels (default: 32)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=100,
        help="Minimum area in pixels (default: 100)",
    )
    parser.add_argument(
        "--merge-distance",
        type=float,
        default=10.0,
        help="Distance in meters to merge nearby polygons (default: 10.0)",
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    print("=" * 80)
    print("SINGLE IMAGE INFERENCE")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Image: {args.image}")
    print(f"Device: {device}")
    print(f"Tile size: {args.tile_size}x{args.tile_size}")
    print(f"Overlap: {args.overlap} pixels")
    print("=" * 80)
    print()

    # Load model
    print("Loading model...")
    model = UNetSegmentor.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.to(device)
    print("✓ Model loaded")

    # Run inference
    print(f"\nProcessing image: {args.image}")
    prediction_mask, transform, crs = predict_single_image(
        model, args.image, device, tile_size=args.tile_size, overlap=args.overlap
    )

    # Count predictions
    datacenter_pixels = prediction_mask.sum()
    total_pixels = prediction_mask.size
    print(f"\n✓ Prediction complete")
    print(
        f"  Datacenter pixels: {datacenter_pixels:,} ({datacenter_pixels / total_pixels * 100:.2f}%)"
    )

    # Save mask if requested
    if args.output_mask:
        with rasterio.open(args.image) as src:
            profile = src.profile.copy()
            profile.update(dtype=rasterio.uint8, count=1, compress="lzw")

        output_mask_path = Path(args.output_mask)
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(output_mask_path, "w", **profile) as dst:
            dst.write(prediction_mask, 1)

        print(f"✓ Mask saved to: {args.output_mask}")

    # Extract centroids
    print(
        f"\nExtracting centroids (min area: {args.min_area} pixels, merge distance: {args.merge_distance}m)..."
    )
    centroids = extract_centroids(
        prediction_mask,
        transform,
        crs,
        min_area_pixels=args.min_area,
        merge_distance_m=args.merge_distance,
    )

    if len(centroids) == 0:
        print("⚠ No datacenters detected!")
        return

    print(f"✓ Found {len(centroids)} datacenter(s)")

    # Save centroids
    output_path = Path(args.output_geojson)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    centroids.to_file(output_path, driver="GeoJSON")
    print(f"✓ Centroids saved to: {args.output_geojson}")

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Datacenters detected: {len(centroids)}")
    print(f"Average area: {centroids['area_sqm'].mean():.1f} m²")
    print(f"Total area: {centroids['area_sqm'].sum():.1f} m²")
    print("=" * 80)
    print("\nCentroids:")
    print(centroids[["lat", "lon", "area_sqm"]])
    print("=" * 80)


if __name__ == "__main__":
    main()
