import rasterio
import numpy as np

def apply_cloud_mask_and_normalize(input_tif: str, output_tif: str):
    """
    Applies Sentinel-2 SCL cloud and snow mask, then normalizes reflectance.
    """
    print(f"Processing and masking clouds for: {input_tif}")
    
    with rasterio.open(input_tif) as src:
        bands = src.read()  # Shape: (C, H, W)
        profile = src.profile

    spectral_bands = bands[:4].astype(np.float32)
    scl_band = bands[4]

    # SCL values for clouds, shadows, and snow to mask out
    cloud_snow_classes = {3, 8, 9, 10, 11}
    mask = np.isin(scl_band, list(cloud_snow_classes), invert=True)

    for i in range(spectral_bands.shape[0]):
        spectral_bands[i] = np.where(mask, spectral_bands[i], 0)

    # Normalize reflectance values
    spectral_bands = np.clip(spectral_bands / 10000.0, 0.0, 1.0)

    profile.update({
        'count': 4,
        'dtype': 'float32'
    })

    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(spectral_bands)

    print(f"Cleaned and normalized raster saved to {output_tif}")