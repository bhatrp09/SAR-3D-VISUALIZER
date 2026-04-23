# src/download.py
import os
from dotenv import load_dotenv
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date

load_dotenv()

USERNAME = os.getenv("COPERNICUS_USER")
PASSWORD = os.getenv("COPERNICUS_PASS")

def download_sentinel1(area_wkt, start_date, end_date, output_dir="data/raw"):
    """
    Download Sentinel-1 GRD products for a given area and date range.
    area_wkt: Well-Known Text polygon of your area of interest
    """
    api = SentinelAPI(USERNAME, PASSWORD, "https://scihub.copernicus.eu/dhus")

    products = api.query(
        area=area_wkt,
        date=(start_date, end_date),
        platformname="Sentinel-1",
        producttype="GRD",
        polarisationmode="VV VH",
        sensoroperationalmode="IW"
    )

    print(f"Found {len(products)} products")
    api.download_all(products, directory_path=output_dir)
    return list(products.keys())

if __name__ == "__main__":
    # Bengaluru bounding box (replace with your area)
    wkt = "POLYGON((77.4 12.8, 77.8 12.8, 77.8 13.1, 77.4 13.1, 77.4 12.8))"
    download_sentinel1(wkt, date(2023, 7, 1), date(2023, 7, 31))