<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,20,24&height=200&section=header&text=FemurRecon%20AI&fontSize=48&fontColor=fff&animation=twinkling&fontAlignY=36&desc=3D%20Medical%20Image%20Segmentation%20%26%20Implant%20Planning&descAlignY=55&descSize=17" width="100%"/>

</div>

<div align="center">

![Status](https://img.shields.io/badge/Status-Completed%20%26%20Deployed-brightgreen?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Medical AI](https://img.shields.io/badge/Domain-Medical%20Imaging%20AI-blueviolet?style=for-the-badge)

</div>

---

## 🧠 Overview

**FemurRecon AI** is a deep learning system for automated femur segmentation, fracture classification, and surgical implant size prediction from 3D medical scans. It addresses a critical bottleneck in orthopedic surgery planning — reducing the manual, error-prone process of analyzing CT scans into an automated, reproducible AI pipeline.

> **Core problem solved:** Orthopedic surgeons spend significant time manually measuring femur geometry from CT scans to select implant sizes. FemurRecon AI automates this end-to-end — from raw scan to implant recommendation.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🦴 **3D Femur Segmentation** | Pixel-accurate volumetric segmentation using 3D U-Net |
| 🔬 **Fracture Classification** | AO/OTA standard fracture type identification |
| 📐 **Implant Size Prediction** | Automated sizing recommendations from reconstructed geometry |
| 🧊 **3D Reconstruction** | Marching Cubes algorithm for surface mesh generation |
| 🤖 **Synthetic Data Pipeline** | Custom data augmentation to overcome medical data scarcity |
| ⚡ **REST API** | FastAPI deployment for real-time inference |

---

## 🏗️ Architecture

```
CT Scan Input (NIfTI / DICOM)
        │
        ▼
┌─────────────────────────────┐
│   Preprocessing Pipeline    │
│  - Intensity normalization  │
│  - Patch extraction         │
│  - Synthetic augmentation   │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│        3D U-Net             │
│  - Residual conv blocks     │
│  - ASPP (multi-scale ctx)   │
│  - Skip connections         │
│  - Encoder-Decoder          │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│     Post-processing         │
│  - Marching Cubes (mesh)    │
│  - AO/OTA classification    │
│  - Implant size inference   │
└────────────┬────────────────┘
             │
             ▼
     FastAPI REST Endpoint
   (segmentation mask + report)
```

---

## 🔬 Technical Deep Dive

### Model: 3D U-Net with Enhancements

The backbone is a **3D U-Net** — chosen for its proven effectiveness in volumetric medical segmentation. Key architectural additions:

- **Residual Convolution Blocks** — alleviate vanishing gradients in deep 3D networks
- **ASPP (Atrous Spatial Pyramid Pooling)** — captures multi-scale context critical for varying femur sizes across patients
- **Skip Connections** — preserve fine-grained spatial detail lost during downsampling

### The Data Scarcity Problem

Annotated 3D medical scans are extremely limited due to privacy regulations and annotation cost. FemurRecon AI solves this with a **synthetic data generation pipeline**:
- Elastic deformations, random rotations, intensity jitter on existing scans
- Simulated fracture patterns to expand rare fracture-class samples
- Enabled training on a clinically viable model without a large proprietary dataset

### Evaluation Metrics

| Metric | Score |
|---|---|
| **Dice Score** | *(add your value)* |
| **IoU (Intersection over Union)** | *(add your value)* |

### Fracture Classification

Fractures are classified following the **AO/OTA system** — the international standard used by orthopedic surgeons — making the output directly interpretable in clinical contexts.

---

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.8
torch >= 1.12
fastapi
uvicorn
nibabel
scikit-image
numpy
```

### Installation

```bash
git clone https://github.com/Irisrose2000/FemurReconAI.git
cd FemurReconAI
pip install -r requirements.txt
```

### Run the API

```bash
uvicorn app.main:app --reload
```

### Inference Example

```python
import requests

with open("femur_scan.nii.gz", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"scan": f}
    )

result = response.json()
print(result["fracture_type"])      # AO/OTA classification
print(result["implant_size"])       # Recommended implant
print(result["dice_score"])         # Segmentation confidence
```

---

## 📁 Project Structure

```
FemurReconAI/
│
├── app/
│   ├── main.py              # FastAPI app entry point
│   ├── model.py             # 3D U-Net architecture
│   ├── predict.py           # Inference pipeline
│   └── utils.py             # Preprocessing & Marching Cubes
│
├── training/
│   ├── train.py             # Training loop
│   ├── dataset.py           # Data loader + augmentation
│   └── losses.py            # Dice + Cross-entropy loss
│
├── synthetic_data/
│   └── generator.py         # Synthetic augmentation pipeline
│
├── notebooks/
│   └── exploration.ipynb    # EDA and result visualization
│
├── requirements.txt
└── README.md
```

---

## 🌍 Real-World Impact

- Reduces pre-surgical planning time for orthopedic procedures
- Standardizes implant sizing — reducing human measurement error
- AO/OTA-aligned output means results are immediately usable by surgeons
- Synthetic data pipeline is reusable for other scarce medical imaging domains

---

## 👩‍💻 Author

**Aleena Johnson**
B.Tech AI & Data Science | Thrissur, Kerala

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/aleena-johnson-7639a9282)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat-square&logo=github&logoColor=white)](https://github.com/Irisrose2000)

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,20,24&height=100&section=footer" width="100%"/>

</div>
