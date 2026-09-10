import os
import rasterio
from rasterio.windows import Window

def extract_patches(input_tif: str, output_dir: str, patch_size: int = 256):
    print(f"Tiling {input_tif} into {patch_size}x{patch_size} patches...")
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(input_tif) as src:
        width = src.width
        height = src.height
        base_transform = src.transform

        patch_count = 0
        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                w = min(patch_size, width - x)
                h = min(patch_size, height - y)

                if w < patch_size or h < patch_size:
                    continue

                window = Window(x, y, w, h)
                patch_data = src.read(window=window)

                transform = rasterio.windows.transform(window, base_transform)
                patch_profile = src.profile.copy()
                patch_profile.update({
                    'height': h,
                    'width': w,
                    'transform': transform
                })

                patch_filename = os.path.join(output_dir, f"patch_x{x}_y{y}.tif")
                with rasterio.open(patch_filename, 'w', **patch_profile) as dst:
                    dst.write(patch_data)

                patch_count += 1

    print(f"Generated {patch_count} patches successfully in '{output_dir}' directory.")