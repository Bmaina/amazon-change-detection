"""
01_download_data.py
===================
Downloads two Sentinel-2 scenes of the same Amazon location
from different years using the Copernicus Data Space Ecosystem (CDSE) API.

No account required for small downloads.
Scenes downloaded: Rondônia, Brazil — tile 21LYG
  - 2019 (pre-deforestation baseline)
  - 2024 (post-deforestation)

Usage:
    python 01_download_data.py

Output:
    data/raw/2019/  — Sentinel-2 bands for 2019 scene
    data/raw/2024/  — Sentinel-2 bands for 2024 scene
"""

import os
import requests
import zipfile
import shutil
from pathlib import Path
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────────

# Rondônia, Brazil bounding box (lon_min, lat_min, lon_max, lat_max)
# This is the "fishbone" deforestation corridor — one of the most dramatic
# deforestation zones on Earth, visible from space
BBOX = (-63.5, -11.5, -62.5, -10.5)

# Date ranges — pick cloud-free dry season windows (June–September)
DATE_RANGES = {
    "2019": ("2019-07-01", "2019-09-30"),
    "2024": ("2024-07-01", "2024-09-30"),
}

# Sentinel-2 bands to download (RGB + NIR for NDVI)
# B04=Red, B03=Green, B02=Blue, B08=NIR
BANDS = ["B04", "B03", "B02", "B08"]

MAX_CLOUD_COVER = 10   # percent
OUTPUT_DIR = Path("data/raw")

# ── CDSE OpenSearch API (no authentication required) ──────────────────────────

OPENSEARCH_URL = "https://catalogue.dataspace.copernicus.eu/resto/api/collections/Sentinel2/search.json"
DOWNLOAD_BASE  = "https://zipper.dataspace.copernicus.eu/zip"


def search_scenes(year, date_start, date_end):
    """Search for Sentinel-2 L2A scenes over the AOI."""
    lon_min, lat_min, lon_max, lat_max = BBOX
    params = {
        "startDate":        date_start,
        "completionDate":   date_end,
        "processingLevel": "S2MSI2A",
        "cloudCover":      f"[0,{MAX_CLOUD_COVER}]",
        "box":             f"{lon_min},{lat_min},{lon_max},{lat_max}",
        "maxRecords":      5,
        "sortParam":       "cloudCover",
        "sortOrder":       "ascending",
    }
    print(f"\n🔍 Searching for {year} scenes...")
    resp = requests.get(OPENSEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    features = resp.json().get("features", [])
    print(f"   Found {len(features)} scene(s) with < {MAX_CLOUD_COVER}% cloud cover")
    return features


def download_scene(scene, year):
    """Download and extract a scene's RGB+NIR bands."""
    out_dir = OUTPUT_DIR / year
    out_dir.mkdir(parents=True, exist_ok=True)

    product_id   = scene["id"]
    product_name = scene["properties"]["title"]
    cloud_cover  = scene["properties"].get("cloudCover", "?")
    date         = scene["properties"].get("startDate", "?")[:10]

    print(f"\n📥 Downloading {year} scene:")
    print(f"   Name        : {product_name}")
    print(f"   Date        : {date}")
    print(f"   Cloud cover : {cloud_cover}%")
    print(f"   Product ID  : {product_id}")

    # Download zip via CDSE zipper
    zip_url  = f"{DOWNLOAD_BASE}?productId={product_id}"
    zip_path = out_dir / f"{year}_scene.zip"

    print(f"   Downloading to {zip_path} ...")
    with requests.get(zip_url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r   Progress: {pct:.1f}%", end="", flush=True)
    print()

    # Extract only the bands we need
    print(f"   Extracting bands: {BANDS}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if any(f"_{band}_" in member or member.endswith(f"_{band}.jp2")
                   for band in BANDS):
                zf.extract(member, out_dir)
                print(f"   Extracted: {Path(member).name}")

    zip_path.unlink()   # remove zip to save space
    print(f"   ✅ {year} scene ready in {out_dir}")
    return out_dir


def download_manual_fallback():
    """
    If the API download fails (authentication required for large files),
    print step-by-step instructions for manual download via browser.
    """
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║           MANUAL DOWNLOAD INSTRUCTIONS (if API fails)                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  1. Go to: https://browser.dataspace.copernicus.eu                       ║
║  2. Create a free account (takes 2 minutes)                              ║
║  3. Search for location: Rondônia, Brazil                                ║
║     Coordinates: -63.5, -11.5, -62.5, -10.5                             ║
║  4. Set filters:                                                          ║
║     - Data source : Sentinel-2 L2A                                       ║
║     - Cloud cover : 0–10%                                                ║
║     - Date range  : 2019-07-01 to 2019-09-30  (for 2019 scene)          ║
║  5. Click the best result → Download (select "All Bands")                ║
║  6. Repeat steps 4–5 for date range: 2024-07-01 to 2024-09-30           ║
║  7. Extract downloaded zips to:                                           ║
║     data/raw/2019/   and   data/raw/2024/                                ║
║                                                                          ║
║  Alternative: Use Google Earth Engine (see README for GEE script)        ║
╚══════════════════════════════════════════════════════════════════════════╝
""")


def use_sample_data():
    """
    Download small sample patches from a public GeoTIFF source
    so the pipeline can be tested immediately without a large download.
    Uses pre-clipped Amazon Sentinel-2 scenes hosted on GitHub.
    """
    SAMPLE_URLS = {
        "2019": "https://github.com/Bmaina/dofa-eurosat-segmentation/releases/download/v1.0/amazon_2019_sample.tif",
        "2024": "https://github.com/Bmaina/dofa-eurosat-segmentation/releases/download/v1.0/amazon_2024_sample.tif",
    }
    print("\n📦 Downloading sample patches for quick demo...")
    for year, url in SAMPLE_URLS.items():
        out_path = OUTPUT_DIR / year / f"amazon_{year}_sample.tif"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            out_path.write_bytes(resp.content)
            print(f"   ✅ {year} sample saved to {out_path}")
        except Exception as e:
            print(f"   ⚠️  Could not download sample for {year}: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  🛰️  Amazon Deforestation — Sentinel-2 Data Download")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    success = True
    for year, (start, end) in DATE_RANGES.items():
        try:
            scenes = search_scenes(year, start, end)
            if not scenes:
                print(f"   ⚠️  No scenes found for {year}. Try widening the date range.")
                success = False
                continue
            download_scene(scenes[0], year)   # take the lowest cloud cover scene
        except requests.HTTPError as e:
            if e.response.status_code in (401, 403):
                print(f"\n⚠️  Authentication required for {year} download.")
                success = False
            else:
                print(f"\n❌ HTTP error for {year}: {e}")
                success = False
        except Exception as e:
            print(f"\n❌ Unexpected error for {year}: {e}")
            success = False

    if not success:
        download_manual_fallback()

    print("\n✅ Data download complete. Run next: python 02_prepare_scenes.py")
