# Traffic Sign Recognition for Varying Lighting Conditions

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-purple)
![License](https://img.shields.io/badge/License-MIT-green)

**Author:** Samson Connelly  
**Institution:** Indiana University - Luddy School of Informatics, Computing and Engineering

## 📖 Overview

This project implements a robust **Traffic Sign Recognition (TSR)** system designed to handle challenging lighting environments (e.g., very dark, very bright, or varying weather).

Standard object detection models often struggle when lighting conditions drift significantly from the training distribution. This project addresses that challenge by using **unsupervised clustering** to categorize images based on their lighting characteristics and utilizing a **mixture of experts** approach—selecting a specialized YOLOv8 model trained specifically for that lighting condition.

### Key Features
* **Unsupervised Clustering:** Uses **Fuzzy C-Means (FCM)** on HSV histograms to group images by lighting/brightness.
* **Dimensionality Reduction:** Applies **Principal Component Analysis (PCA)** to reduce histogram noise.
* **Adaptive Inference:** Dynamically selects the best model for a given image based on its cluster membership probability.
* **Ambiguity Handling:** Includes a fallback "General" model for images that do not strongly belong to any specific lighting cluster.

---

## ⚙️ Methodology

The system follows a two-stage pipeline:

### 1. Clustering & Training Phase
1.  **Preprocessing:** Convert images to **HSV** color space and extract histograms from the **V (Value)** channel.
2.  **PCA:** Reduce histogram features from 32 to 16 components to decrease noise.
3.  **Fuzzy C-Means:** Cluster data into 4 distinct lighting groups:
    * *Cluster 0:* Slightly Dark
    * *Cluster 1:* Very Dark
    * *Cluster 2:* Very Light
    * *Cluster 3:* Slightly Light
4.  **Training:** Train separate YOLOv8 models for each cluster, plus one "General" model on the full dataset.

### 2. Inference Phase
1.  New image $\rightarrow$ HSV Histogram $\rightarrow$ PCA Projection.
2.  Calculate membership probability for all clusters.
3.  **Routing Logic:**
    * If `max_prob < 0.6` $\rightarrow$ Use **General Model** (Ambiguous).
    * If `Cluster 1` (Very Dark) $\rightarrow$ Use **Cluster 0 Model** (due to data scarcity).
    * If `Cluster 2` (Very Light) $\rightarrow$ Use **Cluster 3 Model**.
    * Otherwise $\rightarrow$ Use the specific cluster model.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* [Ultralytics YOLOv8](https://docs.ultralytics.com/)
* Scikit-learn
* OpenCV
* NumPy

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/sbconnelly2000/Traffic-Sign-Recognition-for-Varying-Lighting-Conditions.git](https://github.com/sbconnelly2000/Traffic-Sign-Recognition-for-Varying-Lighting-Conditions.git)
    cd Traffic-Sign-Recognition-for-Varying-Lighting-Conditions
    ```

2.  **Install dependencies:**
    ```bash
    pip install ultralytics scikit-learn opencv-python numpy matplotlib
    ```

---

## 📂 Repository Structure

```text
├── data/                  # Dataset files (images and labels)
├── models/                # Trained YOLOv8 weights (.pt) and PCA/FCM models
├── scripts/
│   ├── clustering.py      # HSV extraction, PCA, and Fuzzy C-Means logic
│   ├── train_pipeline.py  # Data splitting and YOLOv8 training script
│   └── inference.py       # Main prediction pipeline
├── research_paper.pdf     # Full technical report
└── README.md              # Project documentation
