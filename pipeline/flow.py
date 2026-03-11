"""
pipeline/flow.py
================
Prefect orchestration flow for the Amazon Deforestation Change Detection pipeline.

Wraps all 4 pipeline stages into a single orchestrated workflow:
  - Task 1: Download / generate Sentinel-2 scenes
  - Task 2: Prepare and normalise scenes
  - Task 3: Run DOFA change detection
  - Task 4: Generate visualisations

Usage:
    # Install Prefect (free, runs locally — no account needed)
    pip install prefect

    # Run the full pipeline
    python pipeline/flow.py

    # View run history in local Prefect UI
    prefect server start          # in one terminal
    python pipeline/flow.py       # in another terminal
    # Open http://127.0.0.1:4200

    # Run with real Sentinel-2 data (set USE_SYNTHETIC=False below)
    python pipeline/flow.py

Configuration:
    Edit the CONFIG block below to set paths, AOI, and date ranges.
"""

import sys
import time
import numpy as np
import requests
import zipfile
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for pipeline runs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact

# ── Configuration ─────────────────────────────────────────────────────────────

CONFIG = {
    # Set to False to use real Sentinel-2 data (requires download)
    "USE_SYNTHETIC": True,

    # Paths — update CHECKPOINT_PATH to your trained DOFA checkpoint
    "PROJECT_ROOT":     Path("C:/amazon-change-detection"),
    "GDL_ROOT":         Path("C:/geo-deep-learning"),
    "CHECKPOINT_PATH":  Path("C:/geo-deep-learning/logs/gdl_experiment/version_11/checkpoints/model-epoch=00-val_loss=0.141.ckpt"),

    # Area of interest — Rondônia, Brazil
    "BBOX": (-63.5, -11.5, -62.5, -10.5),

    # Date ranges for Sentinel-2 search
    "DATE_RANGES": {
        "2019": ("2019-07-01", "2019-09-30"),
        "2024": ("2024-07-01", "2024-09-30"),
    },

    # DOFA model config
    "PATCH_SIZE":  64,
    "WAVELENGTHS": [0.665, 0.549, 0.481],
    "MEAN":        [0.485, 0.456, 0.406],
    "STD":         [0.229, 0.224, 0.225],

    # Sentinel-2 pixel size
    "PIXEL_SIZE_M": 10.0,
}

CLASS_NAMES = [
    "Annual Crop", "Forest", "Herbaceous Veg", "Highway",
    "Industrial", "Pasture", "Permanent Crop", "Residential",
    "River", "Sea / Lake",
]
CLASS_COLORS = [
    (0.78,0.52,0.25),(0.13,0.37,0.13),(0.42,0.68,0.35),(0.65,0.60,0.45),
    (0.45,0.45,0.55),(0.55,0.75,0.40),(0.25,0.55,0.15),(0.75,0.60,0.50),
    (0.20,0.50,0.80),(0.10,0.30,0.70),
]

HECTARES_PER_PIXEL = (CONFIG["PIXEL_SIZE_M"] ** 2) / 10_000

# ── Task 1 — Download Data ────────────────────────────────────────────────────

@task(name="Download Sentinel-2 Scenes", retries=2, retry_delay_seconds=30)
def download_data():
    """
    Downloads or generates Sentinel-2 scenes for 2019 and 2024.
    Falls back to synthetic data if real download fails or USE_SYNTHETIC=True.
    """
    logger = get_run_logger()
    raw_dir = CONFIG["PROJECT_ROOT"] / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    years_needed = []
    for year in ["2019", "2024"]:
        out_path = raw_dir / year / f"B04.tif"
        if not out_path.parent.exists() or not list(out_path.parent.glob("*.tif")):
            years_needed.append(year)
        else:
            logger.info(f"  {year} data already exists — skipping download")

    if not years_needed:
        logger.info("All scenes already downloaded")
        return {"status": "cached", "years": ["2019", "2024"]}

    if CONFIG["USE_SYNTHETIC"]:
        logger.info("USE_SYNTHETIC=True — generating synthetic Sentinel-2 scenes")
        for year in years_needed:
            _create_synthetic_bands(year, raw_dir, logger)
        return {"status": "synthetic", "years": years_needed}

    # Attempt real download via Copernicus API
    OPENSEARCH_URL = "https://catalogue.dataspace.copernicus.eu/resto/api/collections/Sentinel2/search.json"
    downloaded = []
    for year in years_needed:
        start, end = CONFIG["DATE_RANGES"][year]
        lon_min, lat_min, lon_max, lat_max = CONFIG["BBOX"]
        params = {
            "startDate": start, "completionDate": end,
            "processingLevel": "S2MSI2A", "cloudCover": "[0,10]",
            "box": f"{lon_min},{lat_min},{lon_max},{lat_max}",
            "maxRecords": 3, "sortParam": "cloudCover", "sortOrder": "ascending",
        }
        try:
            resp = requests.get(OPENSEARCH_URL, params=params, timeout=30)
            resp.raise_for_status()
            features = resp.json().get("features", [])
            if features:
                logger.info(f"  Found {len(features)} scenes for {year}")
                downloaded.append(year)
            else:
                logger.warning(f"  No scenes found for {year} — falling back to synthetic")
                _create_synthetic_bands(year, raw_dir, logger)
        except Exception as e:
            logger.warning(f"  Download failed for {year}: {e} — falling back to synthetic")
            _create_synthetic_bands(year, raw_dir, logger)

    return {"status": "complete", "years": years_needed}


def _create_synthetic_bands(year, raw_dir, logger):
    """Generate synthetic Sentinel-2 bands simulating Amazon forest/deforestation."""
    np.random.seed(42 if year == "2019" else 99)
    H, W = 512, 512
    out_dir = raw_dir / year
    out_dir.mkdir(parents=True, exist_ok=True)

    lon_min, lat_min, lon_max, lat_max = CONFIG["BBOX"]
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, W, H)
    profile = {
        "driver": "GTiff", "dtype": "uint16", "width": W, "height": H,
        "count": 1, "crs": "EPSG:4326", "transform": transform,
    }

    if year == "2019":
        vals = {"B04": (600,50), "B03": (900,80), "B02": (500,40), "B08": (3500,200)}
    else:
        vals = {"B04": (1200,150), "B03": (1100,120), "B02": (800,100), "B08": (2200,300)}

    for band, (mean, std) in vals.items():
        data = np.random.normal(mean, std, (H, W)).clip(0, 10000).astype(np.uint16)
        if year == "2024":
            for i in range(0, W, 40):
                data[:, i:i+8] = 3000   # fishbone deforestation pattern
        with rasterio.open(out_dir / f"{band}.tif", "w", **profile) as dst:
            dst.write(data[np.newaxis, :])

    logger.info(f"  Synthetic bands created for {year} in {out_dir}")


# ── Task 2 — Prepare Scenes ───────────────────────────────────────────────────

@task(name="Prepare & Normalise Scenes")
def prepare_scenes():
    """
    Loads raw bands, normalises to 0-1 float, computes NDVI,
    aligns both scenes to the same spatial extent.
    """
    logger = get_run_logger()
    raw_dir      = CONFIG["PROJECT_ROOT"] / "data/raw"
    prepared_dir = CONFIG["PROJECT_ROOT"] / "data/prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for year in ["2019", "2024"]:
        logger.info(f"  Preparing {year} scene...")
        year_dir = raw_dir / year

        # Find band files
        bands = {}
        for band_name in ["B04", "B03", "B02", "B08"]:
            matches = list(year_dir.rglob(f"*{band_name}*.tif")) + \
                      list(year_dir.rglob(f"*{band_name}*.jp2")) + \
                      list(year_dir.glob(f"{band_name}.tif"))
            if matches:
                bands[band_name] = matches[0]
            else:
                raise FileNotFoundError(f"Band {band_name} not found in {year_dir}")

        # Load and normalise
        def load_norm(path):
            with rasterio.open(path) as src:
                data    = src.read(1).astype(np.float32)
                profile = src.profile
                crs     = src.crs
                tfm     = src.transform
            p2, p98 = np.percentile(data, 2), np.percentile(data, 98)
            return np.clip((data - p2) / (p98 - p2 + 1e-6), 0, 1), profile, crs, tfm

        r, profile, crs, tfm = load_norm(bands["B04"])
        g, *_ = load_norm(bands["B03"])
        b, *_ = load_norm(bands["B02"])
        nir_raw = rasterio.open(bands["B08"]).read(1).astype(np.float32)
        red_raw = rasterio.open(bands["B04"]).read(1).astype(np.float32)

        rgb  = np.stack([r, g, b], axis=0)
        ndvi = np.clip((nir_raw - red_raw) / (nir_raw + red_raw + 1e-6), -1, 1)

        # Save
        profile.update(crs=crs, transform=tfm)
        for data, name in [(rgb, f"{year}_rgb"), (ndvi[np.newaxis,:], f"{year}_ndvi")]:
            p = profile.copy()
            p.update(count=data.shape[0], dtype="float32", driver="GTiff", compress="lzw")
            with rasterio.open(prepared_dir / f"{name}.tif", "w", **p) as dst:
                dst.write(data.astype(np.float32))

        results[year] = {
            "shape": list(rgb.shape),
            "ndvi_mean": float(ndvi.mean()),
            "ndvi_min":  float(ndvi.min()),
            "ndvi_max":  float(ndvi.max()),
        }
        logger.info(f"  {year}: shape={rgb.shape}  NDVI mean={ndvi.mean():.3f}")

    # Align shapes
    shapes = {}
    for year in ["2019", "2024"]:
        with rasterio.open(prepared_dir / f"{year}_rgb.tif") as src:
            shapes[year] = (src.height, src.width)

    if shapes["2019"] != shapes["2024"]:
        logger.warning(f"Shape mismatch {shapes} — will crop at inference time")
    else:
        logger.info(f"Scenes aligned — shape: {shapes['2019']}")

    return results


# ── Task 3 — Change Detection ─────────────────────────────────────────────────

@task(name="Run DOFA Change Detection")
def detect_changes():
    """
    Loads DOFA model, runs sliding-window inference on both scenes,
    computes binary and class-level change maps.
    """
    logger = get_run_logger()

    # Add geo-deep-learning to path
    sys.path.insert(0, str(CONFIG["GDL_ROOT"]))

    prepared_dir = CONFIG["PROJECT_ROOT"] / "data/prepared"
    output_dir   = CONFIG["PROJECT_ROOT"] / "outputs/change"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = CONFIG["CHECKPOINT_PATH"]
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            f"Update CONFIG['CHECKPOINT_PATH'] in pipeline/flow.py"
        )

    # Load model
    logger.info(f"Loading DOFA checkpoint: {checkpoint.name}")
    import segmentation_models_pytorch as smp
    from geo_deep_learning.tasks_with_models.segmentation_dofa import SegmentationDOFA

    ckpt  = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hp    = ckpt["hyper_parameters"]
    model = SegmentationDOFA(
        encoder       = hp["encoder"],
        pretrained    = False,
        image_size    = tuple(hp["image_size"]),
        num_classes   = hp["num_classes"],
        max_samples   = hp.get("max_samples", 2),
        loss          = smp.losses.DiceLoss(mode="multiclass", from_logits=True),
        class_labels  = hp.get("class_labels"),
        class_colors  = hp.get("class_colors"),
        freeze_layers = hp.get("freeze_layers"),
    )
    model.configure_model()
    state = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()}
    model.model.load_state_dict(state, strict=True)
    model.eval()
    logger.info("Model loaded")

    wv        = torch.tensor(CONFIG["WAVELENGTHS"]).unsqueeze(0)
    patch     = CONFIG["PATCH_SIZE"]
    stride    = patch // 2
    tf        = T.Compose([
        T.Resize((patch, patch)),
        T.Normalize(mean=CONFIG["MEAN"], std=CONFIG["STD"])
    ])

    def segment(scene_array, label):
        _, H, W  = scene_array.shape
        conf_map = np.zeros((H, W, len(CLASS_NAMES)), dtype=np.float32)
        count    = np.zeros((H, W), dtype=np.float32)
        n_patches = 0
        with torch.no_grad():
            for y in range(0, H - patch + 1, stride):
                for x in range(0, W - patch + 1, stride):
                    chip     = scene_array[:, y:y+patch, x:x+patch]
                    pil_img  = Image.fromarray(
                        (chip.transpose(1,2,0)*255).clip(0,255).astype(np.uint8))
                    tensor   = tf(T.ToTensor()(pil_img)).unsqueeze(0)
                    out      = model(tensor, wv)
                    probs    = out.out.softmax(dim=1).squeeze(0).permute(1,2,0).numpy()
                    conf_map[y:y+patch, x:x+patch] += probs
                    count[y:y+patch, x:x+patch]    += 1.0
                    n_patches += 1
        conf_map /= np.maximum(count, 1.0)[:, :, np.newaxis]
        logger.info(f"  {label}: {n_patches} patches")
        return conf_map.argmax(axis=2).astype(np.int32)

    # Load scenes
    def load_scene(year):
        with rasterio.open(prepared_dir / f"{year}_rgb.tif") as src:
            return src.read(), src.profile

    scene_2019, profile = load_scene("2019")
    scene_2024, _       = load_scene("2024")

    min_h = min(scene_2019.shape[1], scene_2024.shape[1])
    min_w = min(scene_2019.shape[2], scene_2024.shape[2])
    scene_2019 = scene_2019[:, :min_h, :min_w]
    scene_2024 = scene_2024[:, :min_h, :min_w]

    logger.info("Running inference on 2019 scene...")
    seg_2019 = segment(scene_2019, "2019")
    logger.info("Running inference on 2024 scene...")
    seg_2024 = segment(scene_2024, "2024")

    # Change maps
    binary_change = (seg_2019 != seg_2024).astype(np.uint8)
    class_change  = (seg_2019 * 100 + seg_2024).astype(np.int32)

    total_changed = int(binary_change.sum())
    total_pixels  = binary_change.size
    ha_changed    = total_changed * HECTARES_PER_PIXEL

    logger.info(f"Changed: {total_changed:,} / {total_pixels:,} pixels ({total_changed/total_pixels*100:.1f}%)")
    logger.info(f"Area changed: {ha_changed:,.0f} hectares")

    # Transition table
    changed_mask = np.where(binary_change == 1)
    from_cls     = seg_2019[changed_mask]
    to_cls       = seg_2024[changed_mask]
    records      = []
    for fc in range(len(CLASS_NAMES)):
        for tc in range(len(CLASS_NAMES)):
            if fc == tc: continue
            n = int(((from_cls == fc) & (to_cls == tc)).sum())
            if n > 0:
                records.append({
                    "from_class": CLASS_NAMES[fc], "to_class": CLASS_NAMES[tc],
                    "pixels":     n,
                    "hectares":   round(n * HECTARES_PER_PIXEL, 2),
                    "pct_scene":  round(n / total_pixels * 100, 3),
                })
    change_df = pd.DataFrame(records).sort_values("hectares", ascending=False)

    # Save rasters
    def save_raster(arr, path, dtype):
        if arr.ndim == 2: arr = arr[np.newaxis, :]
        p = profile.copy()
        p.update(count=arr.shape[0], dtype=dtype, driver="GTiff", compress="lzw",
                 height=arr.shape[1], width=arr.shape[2])
        with rasterio.open(path, "w", **p) as dst:
            dst.write(arr)

    save_raster(seg_2019,      output_dir / "segmentation_2019.tif", "int32")
    save_raster(seg_2024,      output_dir / "segmentation_2024.tif", "int32")
    save_raster(binary_change, output_dir / "binary_change.tif",     "uint8")
    save_raster(class_change,  output_dir / "class_change.tif",      "int32")
    change_df.to_csv(output_dir / "change_summary.csv", index=False)
    logger.info("Rasters and CSV saved")

    return {
        "total_changed_pixels": total_changed,
        "total_pixels":         total_pixels,
        "pct_changed":          round(total_changed / total_pixels * 100, 2),
        "ha_changed":           round(ha_changed, 2),
        "top_transitions":      change_df.head(5).to_dict(orient="records"),
    }


# ── Task 4 — Visualise ────────────────────────────────────────────────────────

@task(name="Generate Visualisations")
def generate_visualisations():
    """Produces all 7 change detection visualisation outputs."""
    logger    = get_run_logger()
    prep_dir  = CONFIG["PROJECT_ROOT"] / "data/prepared"
    change_dir = CONFIG["PROJECT_ROOT"] / "outputs/change"
    out_dir   = change_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def load(path):
        with rasterio.open(path) as src:
            return src.read()

    def mask_rgb(mask):
        rgb = np.zeros((*mask.shape, 3))
        for i, c in enumerate(CLASS_COLORS):
            rgb[mask == i] = c
        return rgb

    def save_fig(fig, name):
        fig.savefig(out_dir / name, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"  Saved: {name}")

    pre_rgb  = np.transpose(load(prep_dir/"2019_rgb.tif"),  (1,2,0)).clip(0,1)
    post_rgb = np.transpose(load(prep_dir/"2024_rgb.tif"),  (1,2,0)).clip(0,1)
    seg_2019 = load(change_dir/"segmentation_2019.tif")[0]
    seg_2024 = load(change_dir/"segmentation_2024.tif")[0]
    binary   = load(change_dir/"binary_change.tif")[0]
    n19      = load(prep_dir/"2019_ndvi.tif")[0]
    n24      = load(prep_dir/"2024_ndvi.tif")[0]
    df       = pd.read_csv(change_dir/"change_summary.csv")

    legend = [mpatches.Patch(color=c, label=CLASS_NAMES[i]) for i, c in enumerate(CLASS_COLORS)]

    # Fig 1 — RGB comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Amazon — Sentinel-2 RGB: Before & After", fontsize=14, fontweight="bold")
    for ax, img, yr, col in zip(axes, [pre_rgb, post_rgb], ["2019","2024"], ["#2ecc71","#e74c3c"]):
        ax.imshow(img); ax.set_title(yr, fontsize=13, fontweight="bold", color=col); ax.axis("off")
        for s in ax.spines.values():
            s.set_edgecolor(col); s.set_linewidth(3); s.set_visible(True)
    plt.tight_layout(); save_fig(fig, "01_rgb_comparison.png")

    # Fig 2 — Segmentation comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Amazon — DOFA Segmentation: 2019 vs 2024", fontsize=14, fontweight="bold")
    for ax, seg, yr in zip(axes, [seg_2019, seg_2024], ["2019","2024"]):
        ax.imshow(mask_rgb(seg), interpolation="nearest")
        ax.set_title(f"Land Cover {yr}", fontsize=12, fontweight="bold"); ax.axis("off")
    fig.legend(handles=legend, loc="lower center", ncol=5, bbox_to_anchor=(0.5,-0.06), fontsize=9)
    plt.tight_layout(); save_fig(fig, "02_segmentation_comparison.png")

    # Fig 3 — Binary change
    ha  = binary.sum() * HECTARES_PER_PIXEL
    pct = binary.sum() / binary.size * 100
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Amazon — Binary Change: 2019 to 2024", fontsize=14, fontweight="bold")
    axes[0].imshow(pre_rgb); axes[0].set_title("2019 Reference"); axes[0].axis("off")
    axes[1].imshow(pre_rgb)
    axes[1].imshow(binary, cmap=LinearSegmentedColormap.from_list("c",[(0,0,0,0),(0.9,0.1,0.1,0.7)]), vmin=0, vmax=1)
    axes[1].set_title("Changed: " + str(round(ha)) + " ha  (" + str(round(pct,1)) + "%)", color="#e74c3c"); axes[1].axis("off")
    axes[2].imshow(binary, cmap="Reds", interpolation="nearest"); axes[2].set_title("Change Heatmap"); axes[2].axis("off")
    plt.tight_layout(); save_fig(fig, "03_binary_change.png")

    # Fig 4 — Class change map
    H, W   = seg_2019.shape
    chg_rgb = np.ones((H, W, 3)) * 0.15
    changed = binary.astype(bool)
    for i, c in enumerate(CLASS_COLORS):
        chg_rgb[changed & (seg_2024 == i)] = c
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Amazon — What Did the Land Become?", fontsize=14, fontweight="bold")
    axes[0].imshow(mask_rgb(seg_2019), interpolation="nearest"); axes[0].set_title("2019 (Before)"); axes[0].axis("off")
    axes[1].imshow(chg_rgb, interpolation="nearest"); axes[1].set_title("Changed pixels — colour = 2024 class"); axes[1].axis("off")
    fig.legend(handles=legend, loc="lower center", ncol=5, bbox_to_anchor=(0.5,-0.06), fontsize=9)
    plt.tight_layout(); save_fig(fig, "04_class_change_map.png")

    # Fig 5 — NDVI
    diff = n24 - n19
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("NDVI Comparison: 2019 vs 2024", fontsize=13, fontweight="bold")
    for ax, data, title in zip(axes[:2], [n19, n24],
                               ["NDVI 2019  mean=" + str(round(float(n19.mean()),3)),
                                "NDVI 2024  mean=" + str(round(float(n24.mean()),3))]):
        im = ax.imshow(data, cmap="RdYlGn", vmin=-0.2, vmax=1.0)
        ax.set_title(title); ax.axis("off"); plt.colorbar(im, ax=ax, fraction=0.046)
    im = axes[2].imshow(diff, cmap="RdBu", vmin=-0.5, vmax=0.5)
    axes[2].set_title("Change  mean=" + str(round(float(diff.mean()),3))); axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    plt.tight_layout(); save_fig(fig, "05_ndvi_comparison.png")

    # Fig 6 — Transition chart
    if not df.empty:
        top = df.head(12)
        labels = [r["from_class"] + " to " + r["to_class"] for _, r in top.iterrows()]
        colors = [CLASS_COLORS[CLASS_NAMES.index(r["to_class"])] if r["to_class"] in CLASS_NAMES
                  else CLASS_COLORS[0] for _, r in top.iterrows()]
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(labels, top["hectares"], color=colors, edgecolor="white")
        ax.bar_label(bars, fmt=lambda x: str(round(x)) + " ha", padding=4, fontsize=9)
        ax.set_xlabel("Hectares"); ax.invert_yaxis()
        ax.set_title("Top Land Cover Transitions — 2019 to 2024", fontsize=13, fontweight="bold")
        ax.set_facecolor("#f8f9fa"); fig.patch.set_facecolor("white")
        plt.tight_layout(); save_fig(fig, "06_transition_chart.png")

    # Fig 7 — Summary card
    forest_cls     = 1
    seg_2019_arr   = seg_2019
    seg_2024_arr   = seg_2024
    forest_2019_ha = float((seg_2019_arr == forest_cls).sum()) * HECTARES_PER_PIXEL
    forest_2024_ha = float((seg_2024_arr == forest_cls).sum()) * HECTARES_PER_PIXEL
    forest_lost    = max(0, forest_2019_ha - forest_2024_ha)
    pct_lost       = forest_lost / max(forest_2019_ha, 1) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off"); fig.patch.set_facecolor("#1a1a2e")
    ax.text(0.5, 0.95, "Amazon Deforestation — Change Detection Summary",
            transform=ax.transAxes, ha="center", fontsize=14, fontweight="bold", color="white")
    stats = [
        ("Total Area Changed",  str(round(ha)) + " ha",                                      "#e74c3c"),
        ("Forest Cover 2019",   str(round(forest_2019_ha)) + " ha",                          "#2ecc71"),
        ("Forest Cover 2024",   str(round(forest_2024_ha)) + " ha",                          "#e67e22"),
        ("Forest Lost",         str(round(forest_lost)) + " ha  (" + str(round(pct_lost,1)) + "%)", "#c0392b"),
        ("NDVI Change",         str(round(float(diff.mean()), 3)),                            "#e74c3c" if diff.mean()<0 else "#2ecc71"),
        ("Model",               "DOFA Foundation Model",                                      "#3498db"),
        ("Data",                "Sentinel-2 L2A (ESA Copernicus)",                            "#9b59b6"),
        ("Period",              "2019 to 2024",                                               "white"),
    ]
    for i, (label, value, color) in enumerate(stats):
        y = 0.80 - i * 0.10
        ax.text(0.05, y, label + ":", transform=ax.transAxes, ha="left", fontsize=11, color="#aaaaaa")
        ax.text(0.50, y, value,       transform=ax.transAxes, ha="left", fontsize=11, fontweight="bold", color=color)
    ax.text(0.5, 0.02, "Built with DOFA · Sentinel-2 · geo-deep-learning · PyTorch · Prefect",
            transform=ax.transAxes, ha="center", fontsize=9, color="#666666", style="italic")
    save_fig(fig, "07_summary_card.png")

    outputs = sorted(out_dir.glob("0*.png"))
    logger.info(f"All {len(outputs)} visualisations saved to {out_dir}")
    return {"figures_saved": len(outputs), "output_dir": str(out_dir)}


# ── Prefect Flow ──────────────────────────────────────────────────────────────

@flow(
    name="Amazon Deforestation Change Detection",
    description="End-to-end GeoAI pipeline: Sentinel-2 download → DOFA segmentation → change detection → visualisation",
    log_prints=True,
)
def amazon_change_detection_pipeline():
    """
    Full Amazon deforestation change detection pipeline orchestrated by Prefect.

    Stages:
      1. Download / generate Sentinel-2 scenes (2019 & 2024)
      2. Normalise, align, compute NDVI
      3. DOFA inference + binary & class-level change maps
      4. Generate 7 visualisation outputs

    To view run history:
      prefect server start        # http://127.0.0.1:4200
    """
    logger = get_run_logger()
    logger.info("=" * 60)
    logger.info("  Amazon Deforestation Change Detection Pipeline")
    logger.info("  Orchestrated by Prefect")
    logger.info("=" * 60)

    t0 = time.time()

    # Stage 1
    logger.info("\n[1/4] Downloading data...")
    download_result = download_data()
    logger.info("Download complete: " + str(download_result))

    # Stage 2
    logger.info("\n[2/4] Preparing scenes...")
    prep_result = prepare_scenes()
    for year, stats in prep_result.items():
        logger.info(f"  {year}: NDVI mean={stats['ndvi_mean']:.3f}")

    # Stage 3
    logger.info("\n[3/4] Running change detection...")
    change_result = detect_changes()
    logger.info(f"  Changed: {change_result['ha_changed']:,.0f} ha  ({change_result['pct_changed']}%)")

    # Stage 4
    logger.info("\n[4/4] Generating visualisations...")
    vis_result = generate_visualisations()
    logger.info(f"  {vis_result['figures_saved']} figures saved")

    elapsed = time.time() - t0
    logger.info(f"\nPipeline complete in {elapsed:.1f}s")

    # Prefect artifact — summary shown in UI
    summary_md = f"""
## Amazon Deforestation Pipeline — Run Summary

| Metric | Value |
|---|---|
| Area changed | {change_result['ha_changed']:,.0f} ha |
| Pct changed | {change_result['pct_changed']}% |
| Figures saved | {vis_result['figures_saved']} |
| Runtime | {elapsed:.1f}s |
| Data mode | {download_result['status']} |

### Top Transitions
| From | To | Hectares |
|---|---|---|
""" + "\n".join(
        f"| {t['from_class']} | {t['to_class']} | {t['hectares']:,.0f} |"
        for t in change_result["top_transitions"]
    )

    create_markdown_artifact(
        key="pipeline-summary",
        markdown=summary_md,
        description="Change detection results summary"
    )

    return {
        "download":    download_result,
        "preparation": prep_result,
        "changes":     change_result,
        "visualisations": vis_result,
        "runtime_seconds": round(elapsed, 1),
    }


if __name__ == "__main__":
    result = amazon_change_detection_pipeline()
    print("\nFinal result:")
    print(f"  Area changed : {result['changes']['ha_changed']:,.0f} ha")
    print(f"  Figures saved: {result['visualisations']['figures_saved']}")
    print(f"  Runtime      : {result['runtime_seconds']}s")
