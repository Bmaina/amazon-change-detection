"""
04_visualise.py
===============
Produces all visualisation outputs for the Amazon deforestation
change detection pipeline:

  1. Side-by-side RGB: 2019 vs 2024
  2. Side-by-side segmentation maps: 2019 vs 2024
  3. Binary change map (red = changed)
  4. Class-level change map (colour-coded transitions)
  5. NDVI comparison: 2019 vs 2024
  6. Sankey-style transition bar chart (top 10 transitions)
  7. Summary statistics card

Usage:
    python 04_visualise.py

Output:
    outputs/change/01_rgb_comparison.png
    outputs/change/02_segmentation_comparison.png
    outputs/change/03_binary_change.png
    outputs/change/04_class_change_map.png
    outputs/change/05_ndvi_comparison.png
    outputs/change/06_transition_chart.png
    outputs/change/07_summary_card.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import rasterio
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────

PREPARED_DIR = Path("data/prepared")
CHANGE_DIR   = Path("outputs/change")
OUTPUT_DIR   = Path("outputs/change")

CLASS_NAMES = [
    "Annual Crop", "Forest", "Herbaceous Veg", "Highway",
    "Industrial", "Pasture", "Permanent Crop", "Residential",
    "River", "Sea / Lake",
]
CLASS_ICONS = ["🌾", "🌲", "🌿", "🛣️", "🏭", "🐄", "🍇", "🏘️", "🏞️", "🌊"]

CLASS_COLORS = [
    (0.78, 0.52, 0.25),   # Annual Crop   — wheat
    (0.13, 0.37, 0.13),   # Forest        — dark green
    (0.42, 0.68, 0.35),   # Herbaceous    — light green
    (0.65, 0.60, 0.45),   # Highway       — tan
    (0.45, 0.45, 0.55),   # Industrial    — grey-blue
    (0.55, 0.75, 0.40),   # Pasture       — lime green
    (0.25, 0.55, 0.15),   # Permanent Crop— olive green
    (0.75, 0.60, 0.50),   # Residential   — clay
    (0.20, 0.50, 0.80),   # River         — blue
    (0.10, 0.30, 0.70),   # Sea/Lake      — deep blue
]

CMAP = ListedColormap(CLASS_COLORS)
PIXEL_SIZE_M = 10.0
HECTARES_PER_PIXEL = (PIXEL_SIZE_M ** 2) / 10_000


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_raster(path):
    with rasterio.open(path) as src:
        return src.read(), src.profile


def mask_to_rgb(mask):
    rgb = np.zeros((*mask.shape, 3))
    for cls_id, color in enumerate(CLASS_COLORS):
        rgb[mask == cls_id] = color
    return rgb


def make_legend(title=None):
    handles = [
        mpatches.Patch(color=c, label=f"{CLASS_ICONS[i]} {CLASS_NAMES[i]}")
        for i, c in enumerate(CLASS_COLORS)
    ]
    return handles


def save(fig, name):
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"   Saved: {path}")


# ── Plot 1: RGB Side-by-Side ───────────────────────────────────────────────────

def plot_rgb_comparison():
    rgb_2019, _ = load_raster(PREPARED_DIR / "2019_rgb.tif")
    rgb_2024, _ = load_raster(PREPARED_DIR / "2024_rgb.tif")

    # (3, H, W) → (H, W, 3)
    img_2019 = np.transpose(rgb_2019, (1, 2, 0)).clip(0, 1)
    img_2024 = np.transpose(rgb_2024, (1, 2, 0)).clip(0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Amazon — Sentinel-2 RGB: Before & After", fontsize=14, fontweight="bold")

    for ax, img, year, color in zip(axes, [img_2019, img_2024], ["2019", "2024"], ["#2ecc71", "#e74c3c"]):
        ax.imshow(img)
        ax.set_title(f"🛰️ {year}", fontsize=13, fontweight="bold", color=color)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
            spine.set_visible(True)

    plt.tight_layout()
    save(fig, "01_rgb_comparison.png")


# ── Plot 2: Segmentation Side-by-Side ─────────────────────────────────────────

def plot_segmentation_comparison():
    seg_2019, _ = load_raster(CHANGE_DIR / "segmentation_2019.tif")
    seg_2024, _ = load_raster(CHANGE_DIR / "segmentation_2024.tif")

    seg_2019 = seg_2019[0]
    seg_2024 = seg_2024[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Amazon — DOFA Land Cover Segmentation: 2019 vs 2024",
                 fontsize=14, fontweight="bold")

    for ax, seg, year in zip(axes, [seg_2019, seg_2024], ["2019", "2024"]):
        ax.imshow(mask_to_rgb(seg), interpolation="nearest")
        ax.set_title(f"Land Cover {year}", fontsize=12, fontweight="bold")
        ax.axis("off")

    fig.legend(handles=make_legend(), loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.06), fontsize=9, framealpha=0.9)
    plt.tight_layout()
    save(fig, "02_segmentation_comparison.png")


# ── Plot 3: Binary Change Map ──────────────────────────────────────────────────

def plot_binary_change():
    rgb_2019, _   = load_raster(PREPARED_DIR / "2019_rgb.tif")
    binary, _     = load_raster(CHANGE_DIR / "binary_change.tif")

    img_2019 = np.transpose(rgb_2019, (1, 2, 0)).clip(0, 1)
    change   = binary[0]

    total_changed = change.sum()
    ha_changed    = total_changed * HECTARES_PER_PIXEL
    pct_changed   = total_changed / change.size * 100

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Amazon — Binary Change Detection: 2019 → 2024",
                 fontsize=14, fontweight="bold")

    # 2019 reference
    axes[0].imshow(img_2019)
    axes[0].set_title("🛰️ 2019 (Reference)", fontsize=11)
    axes[0].axis("off")

    # Change map overlaid on 2019
    axes[1].imshow(img_2019)
    axes[1].imshow(change, cmap=LinearSegmentedColormap.from_list(
        "change", [(0, 0, 0, 0), (0.9, 0.1, 0.1, 0.7)]), vmin=0, vmax=1)
    axes[1].set_title(f"🔴 Changed Pixels\n{ha_changed:,.0f} ha  ({pct_changed:.1f}% of scene)",
                      fontsize=11, color="#e74c3c")
    axes[1].axis("off")

    # Change-only heatmap
    axes[2].imshow(change, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
    axes[2].set_title("Change Heatmap", fontsize=11)
    axes[2].axis("off")

    plt.tight_layout()
    save(fig, "03_binary_change.png")


# ── Plot 4: Class-Level Change Map ────────────────────────────────────────────

def plot_class_change_map():
    seg_2019, _ = load_raster(CHANGE_DIR / "segmentation_2019.tif")
    seg_2024, _ = load_raster(CHANGE_DIR / "segmentation_2024.tif")
    binary, _   = load_raster(CHANGE_DIR / "binary_change.tif")

    seg_2019 = seg_2019[0]
    seg_2024 = seg_2024[0]
    changed  = binary[0].astype(bool)

    # Highlight only the pixels that changed — colour by new (2024) class
    H, W = seg_2019.shape
    change_rgb = np.ones((H, W, 3)) * 0.15   # dark background = unchanged

    for cls_id, color in enumerate(CLASS_COLORS):
        mask = changed & (seg_2024 == cls_id)
        change_rgb[mask] = color

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Amazon — Class-Level Change: What Did the Land Become?",
                 fontsize=14, fontweight="bold")

    axes[0].imshow(mask_to_rgb(seg_2019), interpolation="nearest")
    axes[0].set_title("Land Cover 2019 (Before)", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(change_rgb, interpolation="nearest")
    axes[1].set_title("Changed Pixels — Coloured by 2024 Class\n(Dark = Unchanged)", fontsize=11)
    axes[1].axis("off")

    fig.legend(handles=make_legend(), loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.06), fontsize=9, framealpha=0.9)
    plt.tight_layout()
    save(fig, "04_class_change_map.png")


# ── Plot 5: NDVI Comparison ───────────────────────────────────────────────────

def plot_ndvi_comparison():
    ndvi_2019, _ = load_raster(PREPARED_DIR / "2019_ndvi.tif")
    ndvi_2024, _ = load_raster(PREPARED_DIR / "2024_ndvi.tif")

    n19 = ndvi_2019[0]
    n24 = ndvi_2024[0]
    diff = n24 - n19    # negative = vegetation loss

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Amazon — NDVI (Vegetation Index): 2019 vs 2024\n"
                 "Higher NDVI = Denser Vegetation | Negative Change = Forest Loss",
                 fontsize=13, fontweight="bold")

    ndvi_cmap = plt.cm.RdYlGn

    im0 = axes[0].imshow(n19, cmap=ndvi_cmap, vmin=-0.2, vmax=1.0)
    axes[0].set_title(f"NDVI 2019\nMean: {n19.mean():.3f}", fontsize=11)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(n24, cmap=ndvi_cmap, vmin=-0.2, vmax=1.0)
    axes[1].set_title(f"NDVI 2024\nMean: {n24.mean():.3f}", fontsize=11)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(diff, cmap="RdBu", vmin=-0.5, vmax=0.5)
    axes[2].set_title(f"NDVI Change (2024 - 2019)\nMean change: {diff.mean():.3f}",
                      fontsize=11, color="#c0392b" if diff.mean() < 0 else "#27ae60")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="NDVI change")

    plt.tight_layout()
    save(fig, "05_ndvi_comparison.png")


# ── Plot 6: Transition Bar Chart ───────────────────────────────────────────────

def plot_transition_chart():
    csv_path = CHANGE_DIR / "change_summary.csv"
    if not csv_path.exists():
        print("   ⚠️  change_summary.csv not found — skipping transition chart")
        return

    df = pd.read_csv(csv_path).head(12)
    if df.empty:
        print("   ⚠️  No transitions found — skipping chart")
        return

    labels = [f"{r['from_class']} → {r['to_class']}" for _, r in df.iterrows()]

    # Colour bars by the destination class
    bar_colors = []
    for _, row in df.iterrows():
        to_idx = CLASS_NAMES.index(row["to_class"]) if row["to_class"] in CLASS_NAMES else 0
        bar_colors.append(CLASS_COLORS[to_idx])

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(labels, df["hectares"], color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fmt=lambda x: f"{x:,.0f} ha", padding=4, fontsize=9)
    ax.set_xlabel("Area (hectares)", fontsize=11)
    ax.set_title("Top Land Cover Transitions — 2019 to 2024\n(Colour = destination class)",
                 fontsize=13, fontweight="bold")
    ax.set_facecolor("#f8f9fa")
    ax.invert_yaxis()
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    save(fig, "06_transition_chart.png")


# ── Plot 7: Summary Card ───────────────────────────────────────────────────────

def plot_summary_card():
    binary, _   = load_raster(CHANGE_DIR / "binary_change.tif")
    seg_2019, _ = load_raster(CHANGE_DIR / "segmentation_2019.tif")
    seg_2024, _ = load_raster(CHANGE_DIR / "segmentation_2024.tif")
    ndvi_2019, _ = load_raster(PREPARED_DIR / "2019_ndvi.tif")
    ndvi_2024, _ = load_raster(PREPARED_DIR / "2024_ndvi.tif")

    change   = binary[0]
    seg19    = seg_2019[0]
    seg24    = seg_2024[0]
    n19      = ndvi_2019[0]
    n24      = ndvi_2024[0]

    forest_cls = 1   # Class index for Forest
    forest_2019_ha = (seg19 == forest_cls).sum() * HECTARES_PER_PIXEL
    forest_2024_ha = (seg24 == forest_cls).sum() * HECTARES_PER_PIXEL
    forest_lost_ha = max(0, forest_2019_ha - forest_2024_ha)
    pct_forest_lost = forest_lost_ha / max(forest_2019_ha, 1) * 100

    total_changed_ha = change.sum() * HECTARES_PER_PIXEL
    ndvi_change      = n24.mean() - n19.mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    fig.patch.set_facecolor("#1a1a2e")

    title = "🌳 Amazon Deforestation — Change Detection Summary"
    ax.text(0.5, 0.95, title, transform=ax.transAxes, ha="center", va="top",
            fontsize=14, fontweight="bold", color="white")

    stats = [
        ("Total Area Changed",     f"{total_changed_ha:,.0f} ha",  "#e74c3c"),
        ("Forest Cover 2019",      f"{forest_2019_ha:,.0f} ha",    "#2ecc71"),
        ("Forest Cover 2024",      f"{forest_2024_ha:,.0f} ha",    "#e67e22"),
        ("Forest Lost",            f"{forest_lost_ha:,.0f} ha  ({pct_forest_lost:.1f}%)", "#c0392b"),
        ("NDVI Change",            f"{ndvi_change:+.3f}",          "#e74c3c" if ndvi_change < 0 else "#2ecc71"),
        ("Model",                  "DOFA Foundation Model",         "#3498db"),
        ("Data Source",            "Sentinel-2 L2A (ESA Copernicus)", "#9b59b6"),
        ("Period",                 "2019 → 2024",                   "white"),
    ]

    for i, (label, value, color) in enumerate(stats):
        y = 0.80 - i * 0.10
        ax.text(0.05, y, f"{label}:", transform=ax.transAxes, ha="left", va="center",
                fontsize=11, color="#aaaaaa")
        ax.text(0.50, y, value, transform=ax.transAxes, ha="left", va="center",
                fontsize=11, fontweight="bold", color=color)

    ax.text(0.5, 0.02,
            "Built with DOFA · Sentinel-2 · geo-deep-learning · PyTorch",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, color="#666666", style="italic")

    save(fig, "07_summary_card.png")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  🎨  Amazon Deforestation — Visualisation")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/7] RGB comparison...")
    plot_rgb_comparison()

    print("[2/7] Segmentation comparison...")
    plot_segmentation_comparison()

    print("[3/7] Binary change map...")
    plot_binary_change()

    print("[4/7] Class-level change map...")
    plot_class_change_map()

    print("[5/7] NDVI comparison...")
    plot_ndvi_comparison()

    print("[6/7] Transition chart...")
    plot_transition_chart()

    print("[7/7] Summary card...")
    plot_summary_card()

    print(f"\n✅ All visualisations saved to {OUTPUT_DIR}/")
    for f in sorted(OUTPUT_DIR.glob("0*.png")):
        print(f"   {f}")
