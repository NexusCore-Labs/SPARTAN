import os
import cubo
import rasterio

def fetch_sentinel_patch(lat: float, lon: float, start_date: str, end_date: str, output_path: str):
    print(f"Fetching Sentinel-2 data around Lat: {lat}, Lon: {lon}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        da = cubo.create(
            lat=lat,
            lon=lon,
            collection="sentinel-2-l2a",
            bands=["B02", "B03", "B04", "B08", "SCL"],
            start_date=start_date,
            end_date=end_date,
            edge_size=512,
            resolution=10,
            query={"eo:cloud_cover": {"lt": 20}}
        )
    except Exception as e:
        raise ConnectionError(f"Failed to connect to STAC API or fetch data cube: {e}")

    if da is None or da.shape[0] == 0:
        raise ValueError("No images found for the given criteria and cloud cover constraint.")

    slice_da = da.isel(time=0)

    profile = {
        'driver': 'GTiff',
        'height': slice_da.shape[1],
        'width': slice_da.shape[2],
        'count': slice_da.shape[0],
        'dtype': str(slice_da.dtype),
        'crs': slice_da.attrs.get("crs", "EPSG:4326"),
        'transform': slice_da.attrs.get("transform")
    }

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(slice_da.values)

    print(f"Successfully saved raw GeoTIFF to {output_path}")
    return output_path