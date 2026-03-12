# 🌳 Amazon Deforestation Change Detection with DOFA \& Sentinel-2

> \*\*End-to-End GeoAI Pipeline for Forest Loss Mapping using Satellite Imagery\*\*

!\[Python](https://img.shields.io/badge/Python-3.12-blue?logo=python) !\[PyTorch](https://img.shields.io/badge/PyTorch-Lightning-purple?logo=pytorch) !\[HuggingFace](https://img.shields.io/badge/🤗-DOFA\_Foundation\_Model-yellow) !\[Sentinel](https://img.shields.io/badge/🌍-Sentinel--2\_EO\_Data-green) !\[Location](https://img.shields.io/badge/📍-Rondônia\_Brazil-red)

\---

## 🌍 What Is This Project?

This pipeline detects **where and how much forest has been lost** in the Amazon between 2019 and 2024 — automatically, using satellite imagery and AI.

It uses the **DOFA** foundation model to segment land cover in two Sentinel-2 scenes of the same location taken five years apart. By comparing the two segmentation maps pixel by pixel, it produces:

* A **binary change map** — exactly which pixels changed
* A **class-level change map** — what each pixel changed *from* and *to* (e.g. Forest → Annual Crop)
* A **quantitative summary** — how many hectares of forest were lost, and what the land became
* **NDVI comparison** — vegetation index change as independent validation

\---

## 🎯 Why Rondônia, Brazil?

Rondônia is one of the most rapidly deforested regions on Earth. Its distinctive **"fishbone" pattern** — access roads cut into the forest with farms extending outward on either side — is visible from space and widely used as a benchmark for deforestation monitoring.

!\[RGB Comparison](outputs/change/01\_rgb\_comparison.png)

\---

## 🔄 How Change Detection Works

```
📅 2019 Scene          📅 2024 Scene
Sentinel-2 RGB    →   Sentinel-2 RGB
      ↓                     ↓
  DOFA Segment         DOFA Segment
      ↓                     ↓
  Land Cover Map 2019   Land Cover Map 2024
              ↓         ↓
          Compare pixel by pixel
              ↓
    Binary Change Map  +  Class-Level Change Map
              ↓
    Hectares lost per class transition
```

!\[Segmentation Comparison](outputs/change/02\_segmentation\_comparison.png)

\---

## 📊 Results

### Binary Change Map

Red pixels = land that changed class between 2019 and 2024.

!\[Binary Change](outputs/change/03\_binary\_change.png)

### Class-Level Change Map

Each changed pixel is coloured by its **new** 2024 class. Dark areas = unchanged.

!\[Class Change](outputs/change/04\_class\_change\_map.png)

### NDVI Change

NDVI (Normalised Difference Vegetation Index) measures vegetation density independently of the model. A falling NDVI confirms real forest loss.

!\[NDVI Comparison](outputs/change/05\_ndvi\_comparison.png)

### Top Land Cover Transitions

!\[Transition Chart](outputs/change/06\_transition\_chart.png)

### Summary

!\[Summary Card](outputs/change/07\_summary\_card.png)

\---

## 🚀 How to Run This Pipeline

### 1\. Clone the repo

```bash
git clone https://github.com/Bmaina/dofa-eurosat-segmentation.git
cd dofa-eurosat-segmentation
```

### 2\. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 3\. Download Sentinel-2 data

```bash
python 01\_download\_data.py
```

This will attempt to auto-download two Sentinel-2 scenes of Rondônia, Brazil (2019 and 2024) via the Copernicus Data Space API. If authentication is required, the script prints step-by-step manual download instructions.

**Manual download (if needed):**

1. Go to [browser.dataspace.copernicus.eu](https://browser.dataspace.copernicus.eu)
2. Create a free ESA account
3. Search: Rondônia, Brazil · Sentinel-2 L2A · Cloud cover < 10%
4. Download one scene from **2019-07-01 to 2019-09-30**
5. Download one scene from **2024-07-01 to 2024-09-30**
6. Extract both into `data/raw/2019/` and `data/raw/2024/`

### 4\. Prepare scenes

```bash
python 02\_prepare\_scenes.py
```

Aligns both scenes to the same spatial extent, normalises pixel values, and computes NDVI. If no real data is found, creates a synthetic demo scene so you can test the full pipeline immediately.

### 5\. Run change detection

```bash
python 03\_detect\_changes.py
```

Runs DOFA inference on both scenes using a sliding patch window, then computes binary and class-level change maps. Outputs GeoTIFFs and a CSV summary of transitions.

> ⚠️ Update `CHECKPOINT\_PATH` in `03\_detect\_changes.py` to point to your trained checkpoint.

### 6\. Visualise results

```bash
python 04\_visualise.py
```

Produces all 7 visualisation outputs saved to `outputs/change/`.



\## 🛰️ SAR + Optical Fusion Extension



An extended pipeline adding \*\*Sentinel-1 SAR\*\* alongside Sentinel-2 optical for cloud-robust deforestation detection.



> The Amazon has 70-80% cloud cover year-round — SAR penetrates clouds and works day/night, making fusion essential for reliable monitoring.



\### What the fusion adds



| Feature | Optical-only | SAR + Optical Fusion |

|---|---|---|

| Cloud coverage | ❌ Blocked | ✅ Penetrates clouds |

| Night operation | ❌ No | ✅ Yes |

| Forest structure | ❌ No | ✅ RVI sensitive to canopy |

| Change confirmation | Single source | ✅ Both sensors must agree |



\### SAR preprocessing applied

\- Lee speckle filter (5×5 window)

\- dB → linear → dB conversion

\- VH/VV ratio (forest vs bare soil discriminator)

\- Radar Vegetation Index (RVI) — sensitive to canopy density



\### Fusion strategy

\- \*\*Optical branch:\*\* DOFA with Sentinel-2 wavelengths \[0.665, 0.549, 0.481 μm]

\- \*\*SAR branch:\*\* DOFA with C-band proxy wavelengths \[0.056 μm]

\- \*\*Confidence fusion:\*\* weighted average (optical=0.65, SAR=0.35)

\- \*\*Confirmed change:\*\* pixels where both sensors independently detect change



\### Notebooks

| Notebook | Description |

|---|---|

| `sar\_optical/01\_download\_sar\_optical.ipynb` | Download Sentinel-1 GRD + Sentinel-2, synthetic fallback |

| `sar\_optical/02\_prepare\_sar\_optical.ipynb` | SAR calibration, speckle filter, RVI, sensor fusion stack |

| `sar\_optical/03\_fusion\_change\_detection.ipynb` | Dual-branch DOFA inference + confidence-weighted fusion |

| `sar\_optical/04\_visualise\_fusion.ipynb` | 6 figures comparing optical vs SAR vs fused results |

\---

## 📁 Project Structure

```
amazon-change-detection/
├── 01\_download\_data.py       # Download Sentinel-2 scenes from Copernicus
├── 02\_prepare\_scenes.py      # Align, normalise, compute NDVI
├── 03\_detect\_changes.py      # DOFA inference + change maps
├── 04\_visualise.py           # All visualisations
├── data/
│   ├── raw/
│   │   ├── 2019/             # Raw Sentinel-2 bands
│   │   └── 2024/             # Raw Sentinel-2 bands
│   └── prepared/
│       ├── 2019\_rgb.tif      # Normalised RGB
│       ├── 2024\_rgb.tif
│       ├── 2019\_ndvi.tif     # NDVI
│       └── 2024\_ndvi.tif
└── outputs/change/
    ├── segmentation\_2019.tif
    ├── segmentation\_2024.tif
    ├── binary\_change.tif
    ├── class\_change.tif
    ├── change\_summary.csv
    └── 0\*\_\*.png              # Visualisations
```

\---

## 🧠 Model

Uses **DOFA** (Dynamic One-For-All), a Vision Transformer foundation model pretrained on millions of satellite images — the same model used in the [EuroSAT segmentation pipeline](https://github.com/Bmaina/dofa-eurosat-segmentation).

The encoder is frozen. Only the decoder was fine-tuned on EuroSAT, then applied here directly — **no retraining required** for the new scene.

This demonstrates a core principle of foundation models: **train once, deploy anywhere.**

\---

## 🌿 Land Cover Classes Detected

|Icon|Class|Relevance to Deforestation|
|-|-|-|
|🌲|Forest|Primary class of interest — loss detected here|
|🌾|Annual Crop|Common destination after clearing|
|🌿|Herbaceous Veg|Early-stage cleared land|
|🐄|Pasture|Cattle ranching — major driver of Amazon deforestation|
|🍇|Permanent Crop|Soy and other cash crops|
|🏘️|Residential|Settlement expansion along access roads|

\---

## 🌍 Real-World Applications

|Application|How This Pipeline Applies|
|-|-|
|🔥 **Deforestation Monitoring**|Detect forest loss in near real-time using Sentinel-2's 5-day revisit|
|☮️ **Conflict \& Peace Operations**|Detect agricultural destruction, displacement, and infrastructure damage|
|🌡️ **Carbon Accounting**|Quantify forest cover change for REDD+ carbon credit verification|
|🌊 **Flood \& Disaster Response**|Apply same pipeline to detect land cover changes after floods or earthquakes|

\---

## 📚 References

* **DOFA Foundation Model** — [huggingface.co/earthflow/DOFA](https://huggingface.co/earthflow/DOFA)
* **geo-deep-learning** — NRCan: [github.com/NRCan/geo-deep-learning](https://github.com/NRCan/geo-deep-learning)
* **Copernicus Data Space** — [dataspace.copernicus.eu](https://dataspace.copernicus.eu)
* **Sentinel-2** — ESA: [sentinel.esa.int](https://sentinel.esa.int)
* **PRODES Deforestation Data** — INPE: [terrabrasilis.dpi.inpe.br](http://terrabrasilis.dpi.inpe.br)

\---

<div align="center">
  <strong>Built by Benson M. Gachaga</strong> — Data Scientist | GeoAI Practitioner | Remote Sensing Specialist<br/>
  MBA · M.S. Geoinformation \& Earth Observation · PMP · Microsoft Certified Power BI<br/><br/>
  <a href="https://linkedin.com/in/bensonmgachaga">LinkedIn</a> \&nbsp;·\&nbsp;
  <a href="https://github.com/Bmaina">GitHub</a>
</div>

