import os
import argparse
from spartan.data.fetcher import fetch_sentinel_patch
from spartan.data.preprocessor import apply_cloud_mask_and_normalize
from spartan.data.tiler import extract_patches

if __name__ == "__main__":
    # Set up command-line argument parser so users can pass custom locations & dates
    parser = argparse.ArgumentParser(description="Run Module 1 Data Acquisition & Preprocessing Pipeline")
    parser.add_argument("--lat", type=float, default=19.0760, help="Latitude of Area of Interest (default: Mumbai)")
    parser.add_argument("--lon", type=float, default=72.8777, help="Longitude of Area of Interest (default: Mumbai)")
    parser.add_argument("--start", type=str, default="2025-11-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2026-03-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--patch-size", type=int, default=256, help="Pixel dimensions for tiling patches")

    args = parser.parse_args()

    print("=== STARTING MODULE 1 PIPELINE ===")
    print(f"Target Location -> Latitude: {args.lat}, Longitude: {args.lon}")
    print(f"Time Range     -> {args.start} to {args.end}")
    
    # Define directory paths where outputs will be saved
    raw_file = "spartan/data/raw/aoi_scene.tif"
    clean_file = "spartan/data/processed/clean_scene.tif"
    patches_folder = "spartan/data/processed/patches/"

    # Ensure target output directories exist locally
    os.makedirs("spartan/data/raw", exist_ok=True)
    os.makedirs("spartan/data/processed/patches", exist_ok=True)

    try:
        # Step 1: Fetch Raw Sentinel-2 Data using dynamic arguments
        print("\n--- STEP 1: Fetching Raw Sentinel-2 Data ---")
        fetch_sentinel_patch(args.lat, args.lon, args.start, args.end, raw_file)

        # Step 2: Apply Cloud Masking & Normalization
        print("\n--- STEP 2: Applying Cloud Masking & Normalization ---")
        apply_cloud_mask_and_normalize(raw_file, clean_file)

        # Step 3: Extract Patches for Model Inference
        print("\n--- STEP 3: Extracting Patches for Model Inference ---")
        extract_patches(clean_file, patches_folder, patch_size=args.patch_size)

        print("\n=== MODULE 1 COMPLETE! Patches are ready for Module 2 team. ===")

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed during execution: {e}")