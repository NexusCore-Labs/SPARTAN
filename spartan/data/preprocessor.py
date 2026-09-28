import rasterio
import numpy as np
import torch

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


def preprocess_array(bands: np.ndarray) -> torch.Tensor:
    """
    Extracts True-Color RGB [B4 (Red), B3 (Green), B2 (Blue)] from Sentinel-2
    and normalizes DN (0 - 10000) to reflectance [0.0, 1.0].
    Returns shape (1, 3, H, W).
    """
    arr = bands.astype(np.float32)

    # 12-band Sentinel-2 L2A: Index 3=B4 (Red), Index 2=B3 (Green), Index 1=B2 (Blue)
    if arr.shape[0] == 12:
        arr = arr[[3, 2, 1], :, :]
    elif arr.shape[0] == 4:
        # 4-band [B2, B3, B4, B8] -> extract [B4, B3, B2], drop NIR
        arr = arr[[2, 1, 0], :, :]
    elif arr.shape[0] >= 3:
        arr = arr[:3, :, :]

    # Normalize Sentinel-2 DN to [0.0, 1.0] reflectance
    if arr.max() > 1.0:
        arr = arr / 10000.0
    arr = np.clip(arr, 0.0, 1.0)

    tensor = torch.from_numpy(arr)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)  # Shape: (1, 3, H, W)
    return tensor


