# src/download_gee.py
import ee
import os

ee.Initialize(project='sar-3d-visualizer')
def export_sentinel1(lon, lat, start, end, output_name="S1_export"):
    """
    Export Sentinel-1 GRD from GEE — already calibrated + terrain corrected.
    lon, lat: center point of your area
    """
    point  = ee.Geometry.Point([lon, lat])
    region = point.buffer(20000).bounds()  # 20km radius

    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(region)
          .filterDate(start, end)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .first())

    for band in ["VV", "VH"]:
        task = ee.batch.Export.image.toDrive(
            image=s1.select(band),
            description=f"{output_name}_{band}",
            folder="SAR_exports",
            scale=10,
            region=region,
            fileFormat="GeoTIFF"
        )
        task.start()
        print(f"Export task started: {output_name}_{band}")

if __name__ == "__main__":
    # Bengaluru center
    export_sentinel1(77.59, 12.97, "2023-07-01", "2023-07-31", "Bengaluru_flood")
    print("Check Google Drive > SAR_exports folder in ~10 minutes")