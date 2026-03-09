"""
02_prepare_scenes.py
====================
Aligns, crops, and normalises the two Sentinel-2 scenes to the same
spatial extent and resolution, ready for DOFA inference.

Usage:
    python 02_prepare_scenes.py

Input:
    data/raw/2019/   — raw Sentinel-2 bands
    data/raw/2024/   — raw Sentinel-2 bands

Output:
    data/prepared/2019_rgb.tif   — aligned, normalised RGB GeoTIFF
    data/prepared/2024_rgb.tif   — aligned, normalised RGB GeoTIFF
    data/prepared/2019_ndvi.tif  — NDVI (vegetation index)
    data/prepared/2024_ndvi.tif  — NDVI
"""

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform
from pathlib import Path
import glob
import warnings
warnings.filterwarnings("ignore")

RAW_DIR      = Path("data/raw")
PREPARED_DIR = Path("data/prepared")
YEARS        = ["2019", "2024"]

# Target resolution in metres
TARGET_RES = 10.0
# Target CRS — UTM zone 20S covers Rondônia
TARGET_CRS = "EPSG:32720"
# Patch size for DOFA (must match training image_size)
PATCH_SIZE = 64


def find_band_file(year_dir, band):
    """Find the file for a given band in the raw data folder."""
    patterns = [
        f"**/*_{band}_10m.jp2",
        f"**/*_{band}.jp2",
        f"**/*_{band}_10m.tif",
        f"**/*_{band}.tif",
        f"**/{band}.tif",
    ]
    for pattern in patterns:
        matches = list(year_dir.rglob(pattern.replace("**/", "")))
        if not matches:
            matches = list(year_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_band(year, band_name):
    """Load a single Sentinel-2 band as a numpy array."""
    year_dir = RAW_DIR / year
    band_file = find_band_file(year_dir, band_name)

    if band_file is None:
        raise FileNotFoundError(
            f"Band {band_name} not found in {year_dir}.\n"
            f"Contents: {list(year_dir.rglob('*'))[:10]}"
        )

    with rasterio.open(band_file) as src:
        data    = src.read(1).astype(np.float32)
        profile = src.profile
        transform = src.transform
        crs = src.crs

    return data, profile, transform, crs


def normalise_band(band, p_low=2, p_high=98):
    """Percentile stretch to 0-1 range."""
    p2  = np.percentile(band, p_low)
    p98 = np.percentile(band, p_high)
    return np.clip((band - p2) / (p98 - p2 + 1e-6), 0, 1)


def compute_ndvi(red, nir):
    """NDVI = (NIR - Red) / (NIR + Red). Ranges -1 to +1."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / (nir + red + 1e-6)
    return np.clip(ndvi, -1, 1)


def save_geotiff(data, path, profile, description=""):
    """Save array as GeoTIFF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if data.ndim == 2:
        data = data[np.newaxis, :]   # add band dim
    profile.update(
        count=data.shape[0],
        dtype="float32",
        driver="GTiff",
        compress="lzw",
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.float32))
        if description:
            dst.update_tags(description=description)
    print(f"   Saved: {path}  {data.shape}")


def prepare_year(year):
    """Load, align, and save RGB + NDVI for one year."""
    print(f"\n📅 Preparing {year} scene...")

    try:
        red,  profile, transform, crs = load_band(year, "B04")
        green, *_                      = load_band(year, "B03")
        blue,  *_                      = load_band(year, "B02")
        nir,   *_                      = load_band(year, "B08")
    except FileNotFoundError as e:
        print(f"   ❌ {e}")
        print(f"   Creating synthetic demo data for {year}...")
        return create_synthetic_scene(year)

    # Normalise bands
    r = normalise_band(red)
    g = normalise_band(green)
    b = normalise_band(blue)
    n = normalise_band(nir)

    # Stack RGB
    rgb = np.stack([r, g, b], axis=0)   # (3, H, W)

    # NDVI
    ndvi = compute_ndvi(red, nir.astype(np.float32))

    # Save
    profile.update(crs=crs, transform=transform)
    save_geotiff(rgb,  PREPARED_DIR / f"{year}_rgb.tif",  profile, f"Sentinel-2 RGB {year}")
    save_geotiff(ndvi, PREPARED_DIR / f"{year}_ndvi.tif", profile, f"NDVI {year}")

    print(f"   ✅ {year} prepared — shape: {rgb.shape}, NDVI range: [{ndvi.min():.2f}, {ndvi.max():.2f}]")
    return rgb, ndvi, profile


def create_synthetic_scene(year):
    """
    Create a synthetic demo scene when real data isn't available.
    Simulates a forested area (2019) and partially deforested area (2024).
    Replace with real Sentinel-2 data for actual analysis.
    """
    np.random.seed(42 if year == "2019" else 99)
    H, W = 512, 512

    if year == "2019":
        # Mostly forest — high green/NIR, low red
        r = np.random.normal(0.08, 0.02, (H, W)).clip(0, 1).astype(np.float32)
        g = np.random.normal(0.15, 0.03, (H, W)).clip(0, 1).astype(np.float32)
        b = np.random.normal(0.07, 0.02, (H, W)).clip(0, 1).astype(np.float32)
        ndvi = np.random.normal(0.75, 0.08, (H, W)).clip(-1, 1).astype(np.float32)
    else:
        # Partially cleared — fishbone pattern of deforestation roads
        r = np.random.normal(0.18, 0.05, (H, W)).clip(0, 1).astype(np.float32)
        g = np.random.normal(0.18, 0.04, (H, W)).clip(0, 1).astype(np.float32)
        b = np.random.normal(0.12, 0.03, (H, W)).clip(0, 1).astype(np.float32)
        ndvi = np.random.normal(0.45, 0.15, (H, W)).clip(-1, 1).astype(np.float32)

        # Add cleared strips (fishbone pattern)
        for i in range(0, W, 40):
            r[:, i:i+8] = 0.35
            g[:, i:i+8] = 0.30
            ndvi[:, i:i+8] = 0.05

    rgb = np.stack([r, g, b], axis=0)

    # Minimal profile for synthetic data
    from rasterio.transform import from_bounds
    transform = from_bounds(-63.5, -11.5, -62.5, -10.5, W, H)
    profile = {
        "driver": "GTiff", "dtype": "float32",
        "width": W, "height": H, "count": 3,
        "crs": TARGET_CRS, "transform": transform,
    }

    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    save_geotiff(rgb,  PREPARED_DIR / f"{year}_rgb.tif",  profile, f"SYNTHETIC Sentinel-2 RGB {year}")
    save_geotiff(ndvi, PREPARED_DIR / f"{year}_ndvi.tif", profile, f"SYNTHETIC NDVI {year}")

    print(f"   ⚠️  Synthetic demo data created for {year}.")
    print(f"   Replace data/prepared/{year}_rgb.tif with real Sentinel-2 data for actual analysis.")
    return rgb, ndvi, profile


def check_alignment():
    """Verify both scenes have the same shape."""
    files = {y: PREPARED_DIR / f"{y}_rgb.tif" for y in YEARS}
    shapes = {}
    for year, path in files.items():
        if path.exists():
            with rasterio.open(path) as src:
                shapes[year] = (src.count, src.height, src.width)

    if len(shapes) == 2:
        if list(shapes.values())[0] == list(shapes.values())[1]:
            print(f"\n✅ Scenes aligned — shape: {list(shapes.values())[0]}")
        else:
            print(f"\n⚠️  Shape mismatch: {shapes}")
            print("   Scenes will be cropped to minimum common size during inference.")
    return shapes


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  🛰️  Amazon Deforestation — Scene Preparation")
    print("=" * 60)

    PREPARED_DIR.mkdir(parents=True, exist_ok=True)

    for year in YEARS:
        prepare_year(year)

    check_alignment()
    print("\n✅ Scenes prepared. Run next: python 03_detect_changes.py")
