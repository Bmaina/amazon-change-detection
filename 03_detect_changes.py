"""
03_detect_changes.py
====================
Runs DOFA segmentation on both prepared scenes and produces:
  1. Binary change map  — changed vs unchanged pixels
  2. Class-level change map — what each pixel changed FROM and TO
  3. Quantitative summary — hectares lost per class transition

Usage:
    python 03_detect_changes.py

Input:
    data/prepared/2019_rgb.tif
    data/prepared/2024_rgb.tif
    logs/gdl_experiment/version_11/checkpoints/model-epoch=00-val_loss=0.141.ckpt

Output:
    outputs/change/binary_change.tif
    outputs/change/class_change.tif
    outputs/change/segmentation_2019.tif
    outputs/change/segmentation_2024.tif
    outputs/change/change_summary.csv
"""

import numpy as np
import torch
import rasterio
from rasterio.transform import from_bounds
import pandas as pd
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(".").resolve()))

# ── Configuration ─────────────────────────────────────────────────────────────

CHECKPOINT_PATH = Path(
    "logs/gdl_experiment/version_11/checkpoints/model-epoch=00-val_loss=0.141.ckpt"
)
PREPARED_DIR = Path("data/prepared")
OUTPUT_DIR   = Path("outputs/change")

# Must match training config
PATCH_SIZE  = 64
WAVELENGTHS = torch.tensor([0.665, 0.549, 0.481])   # Sentinel-2 R, G, B in micrometres
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Pixel size in metres (Sentinel-2 10m resolution)
PIXEL_SIZE_M = 10.0
HECTARES_PER_PIXEL = (PIXEL_SIZE_M ** 2) / 10_000

CLASS_NAMES = [
    "Annual Crop", "Forest", "Herbaceous Veg", "Highway",
    "Industrial", "Pasture", "Permanent Crop", "Residential",
    "River", "Sea / Lake",
]
NUM_CLASSES = len(CLASS_NAMES)

# Change significance threshold — pixels must differ by this many
# class probabilities to be counted as a real change (reduces noise)
CHANGE_THRESHOLD = 0.6


# ── Model Loading ──────────────────────────────────────────────────────────────

def load_model():
    """Load DOFA model from checkpoint."""
    from geo_deep_learning.tasks_with_models.segmentation_dofa import SegmentationDOFA
    from geo_deep_learning.models.segmentation.dofa import DOFASegmentationModel
    import segmentation_models_pytorch as smp

    print(f"📦 Loading model from {CHECKPOINT_PATH}...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    hp   = ckpt["hyper_parameters"]

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
    print("   ✅ Model loaded")
    return model


# ── Inference ──────────────────────────────────────────────────────────────────

def load_scene(year):
    """Load prepared RGB GeoTIFF and return array + profile."""
    path = PREPARED_DIR / f"{year}_rgb.tif"
    with rasterio.open(path) as src:
        data    = src.read()          # (3, H, W)  float32  0-1
        profile = src.profile
    return data, profile


def segment_scene(model, scene_array):
    """
    Run DOFA inference on a full scene by sliding a PATCH_SIZE window.
    Returns a full-scene segmentation map (H, W) with class IDs.
    """
    transform = T.Compose([
        T.Resize((PATCH_SIZE, PATCH_SIZE)),
        T.Normalize(mean=MEAN, std=STD),
    ])

    _, H, W = scene_array.shape
    seg_map  = np.zeros((H, W), dtype=np.int32)
    conf_map = np.zeros((H, W, NUM_CLASSES), dtype=np.float32)

    # Slide patch window across the scene with 50% overlap for smoother results
    stride = PATCH_SIZE // 2
    count  = np.zeros((H, W), dtype=np.float32)

    patches_run = 0
    with torch.no_grad():
        for y in range(0, H - PATCH_SIZE + 1, stride):
            for x in range(0, W - PATCH_SIZE + 1, stride):
                patch = scene_array[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]  # (3, 64, 64)

                # Convert to PIL then apply transform
                patch_pil = Image.fromarray(
                    (patch.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                )
                tensor = transform(T.ToTensor()(patch_pil)).unsqueeze(0)  # (1, 3, 64, 64)
                wv     = WAVELENGTHS.unsqueeze(0)                          # (1, 3)

                output = model(tensor, wv)                                 # namedtuple
                probs  = output.out.softmax(dim=1).squeeze(0)             # (C, 64, 64)
                probs_np = probs.permute(1, 2, 0).numpy()                 # (64, 64, C)

                conf_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += probs_np
                count[y:y+PATCH_SIZE, x:x+PATCH_SIZE]    += 1.0
                patches_run += 1

    # Avoid division by zero at edges
    count = np.maximum(count, 1.0)
    conf_map /= count[:, :, np.newaxis]

    # Class with highest average probability
    seg_map = conf_map.argmax(axis=2).astype(np.int32)

    print(f"   Ran inference on {patches_run} patches — scene shape: ({H}, {W})")
    return seg_map, conf_map


# ── Change Detection ───────────────────────────────────────────────────────────

def compute_binary_change(seg_2019, seg_2024):
    """1 = changed, 0 = unchanged."""
    return (seg_2019 != seg_2024).astype(np.uint8)


def compute_class_change(seg_2019, seg_2024):
    """
    Encode FROM→TO class transitions as a single integer.
    change_code = from_class * 100 + to_class
    e.g. Forest(1) → Annual Crop(0) = 100
    """
    return (seg_2019 * 100 + seg_2024).astype(np.int32)


def summarise_changes(seg_2019, seg_2024, binary_change):
    """Build a DataFrame of all class-to-class transitions with area in hectares."""
    H, W = seg_2019.shape
    records = []

    changed_pixels = np.where(binary_change == 1)
    from_classes   = seg_2019[changed_pixels]
    to_classes     = seg_2024[changed_pixels]

    for from_cls in range(NUM_CLASSES):
        for to_cls in range(NUM_CLASSES):
            if from_cls == to_cls:
                continue
            mask    = (from_classes == from_cls) & (to_classes == to_cls)
            n_pix   = mask.sum()
            if n_pix > 0:
                hectares = n_pix * HECTARES_PER_PIXEL
                records.append({
                    "from_class":    CLASS_NAMES[from_cls],
                    "to_class":      CLASS_NAMES[to_cls],
                    "pixels":        int(n_pix),
                    "hectares":      round(hectares, 2),
                    "pct_of_scene":  round(n_pix / (H * W) * 100, 3),
                })

    df = pd.DataFrame(records).sort_values("hectares", ascending=False)
    return df


def save_raster(array, path, profile, dtype=None):
    """Save array as GeoTIFF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim == 2:
        array = array[np.newaxis, :]
    p = profile.copy()
    p.update(count=array.shape[0], dtype=dtype or str(array.dtype),
             driver="GTiff", compress="lzw")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(array)
    print(f"   Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  🌳  Amazon Deforestation — Change Detection")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model()

    # Load scenes
    print("\n📂 Loading prepared scenes...")
    scene_2019, profile = load_scene("2019")
    scene_2024, _       = load_scene("2024")

    # Ensure same spatial extent (crop to minimum)
    min_h = min(scene_2019.shape[1], scene_2024.shape[1])
    min_w = min(scene_2019.shape[2], scene_2024.shape[2])
    scene_2019 = scene_2019[:, :min_h, :min_w]
    scene_2024 = scene_2024[:, :min_h, :min_w]
    print(f"   Scene shape: {scene_2019.shape}")

    # Run segmentation
    print("\n🔍 Segmenting 2019 scene...")
    seg_2019, conf_2019 = segment_scene(model, scene_2019)

    print("\n🔍 Segmenting 2024 scene...")
    seg_2024, conf_2024 = segment_scene(model, scene_2024)

    # Compute change maps
    print("\n🔄 Computing change maps...")
    binary_change = compute_binary_change(seg_2019, seg_2024)
    class_change  = compute_class_change(seg_2019, seg_2024)

    total_changed = binary_change.sum()
    total_pixels  = binary_change.size
    pct_changed   = total_changed / total_pixels * 100
    ha_changed    = total_changed * HECTARES_PER_PIXEL

    print(f"\n📊 Change Summary:")
    print(f"   Total pixels changed : {total_changed:,} / {total_pixels:,} ({pct_changed:.1f}%)")
    print(f"   Area changed         : {ha_changed:,.0f} hectares")

    # Summarise transitions
    change_df = summarise_changes(seg_2019, seg_2024, binary_change)
    print(f"\n🌳 Top class transitions:")
    print(change_df.head(10).to_string(index=False))

    # Save rasters
    print("\n💾 Saving outputs...")
    save_raster(seg_2019,      OUTPUT_DIR / "segmentation_2019.tif", profile, dtype="int32")
    save_raster(seg_2024,      OUTPUT_DIR / "segmentation_2024.tif", profile, dtype="int32")
    save_raster(binary_change, OUTPUT_DIR / "binary_change.tif",     profile, dtype="uint8")
    save_raster(class_change,  OUTPUT_DIR / "class_change.tif",      profile, dtype="int32")

    # Save CSV
    csv_path = OUTPUT_DIR / "change_summary.csv"
    change_df.to_csv(csv_path, index=False)
    print(f"   Saved: {csv_path}")

    print("\n✅ Change detection complete. Run next: python 04_visualise.py")
