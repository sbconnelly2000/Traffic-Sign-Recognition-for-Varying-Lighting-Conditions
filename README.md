# Traffic Sign Recognition for Varying Lighting Conditions

**Author:** Samson Connelly  
**Institution:** Indiana University - Luddy School of Informatics, Computing and Engineering

## 📌 Project Overview
Traffic Sign Recognition (TSR) is a critical component of autonomous vehicle safety. However, varying lighting and weather conditions often negatively impact detection accuracy. [cite_start]This project proposes a novel pipeline that uses **unsupervised learning** to categorize images based on lighting conditions and dynamically selects a specialized **YOLOv8** object detection model trained for that specific lighting environment[cite: 578, 580, 581].

## 🚀 Key Features
* [cite_start]**Unsupervised Clustering:** Uses Fuzzy C-Means (FCM) to group images into clusters based on lighting characteristics (e.g., dark, light, neutral)[cite: 583].
* [cite_start]**Adaptive Model Selection:** Automatically selects the most appropriate YOLOv8 model for a given image based on its lighting cluster[cite: 584].
* [cite_start]**Robust Preprocessing:** Converts images to HSV color space and utilizes Principal Component Analysis (PCA) for feature extraction[cite: 582, 583].
* [cite_start]**Fallback Mechanism:** Includes an "ambiguous" category that defaults to a general model if the lighting condition is not strongly classified[cite: 652].

## 🛠️ Methodology / Pipeline

The system follows a multi-stage pipeline for both training and inference:

1.  **Preprocessing:**
    * [cite_start]Images are converted from **RGB** to **HSV** color space[cite: 582].
    * [cite_start]A histogram is generated based on the **V (Value/Brightness)** channel to capture lighting information[cite: 647].
2.  **Dimensionality Reduction:**
    * [cite_start]**PCA (Principal Component Analysis)** is applied to the histograms to reduce components from 32 to 16, decreasing noise and complexity[cite: 648].
3.  **Clustering (Lighting Classification):**
    * [cite_start]**Fuzzy C-Means (FCM)** clustering classifies the image into one of 4 distinct lighting clusters[cite: 659].
    * [cite_start]*Thresholding:* If the highest cluster probability is **< 0.6**, the image is labeled "Ambiguous" and sent to the General Model[cite: 652].
4.  **Object Detection:**
    * Specific **YOLOv8** models are trained on each valid cluster.
    * [cite_start]**Inference:** The system calculates the image's cluster, selects the corresponding YOLOv8 model (or the General model), and runs detection[cite: 709, 710].

[cite_start]*(Note: Due to data limitations in the study, Clusters 1 and 2 currently route to the models for Clusters 0 and 3 respectively[cite: 756].)*

## 📂 Dataset
* [cite_start]**Source:** Self Driving Cars Dataset (hosted on Roboflow)[cite: 623].
* [cite_start]**Size:** 5,609 annotated images of traffic signs and stop lights[cite: 624].
* [cite_start]**Format:** Resized to 416x416 without augmentation[cite: 625].

## 📊 Results
The unsupervised clustering successfully identified distinct lighting patterns:
* **Cluster 0:** Slightly Dark
* **Cluster 1:** Very Dark
* **Cluster 2:** Very Light
* **Cluster 3:** Slightly Light

[cite_start]While the dataset size limited the conclusiveness of the study, specific clusters (like Cluster 0 and 3) achieved mAP scores for Stop Signs comparable to the general baseline model[cite: 713, 722].

## 💻 Usage

### Prerequisites
* Python 3.x
* Ultralytics (YOLOv8)
* NumPy
* Scikit-Learn (for PCA/Clustering)
* OpenCV

### Running Inference
(Conceptual usage based on paper methodology)

```python
import numpy as np
from ultralytics import YOLO

# 1. Load Precomputed PCA and FCM Centers
pca_model = load_pca("pca_model.pkl")
fcm_centers = np.load("fcm_centers.npy")

# 2. Process Input Image
# Convert to HSV -> Calculate V-Histogram -> Apply PCA
processed_features = preprocess_image(input_image)

# 3. Calculate Cluster Probabilities
probs = calculate_fcm_probabilities(processed_features, fcm_centers)
max_prob = np.max(probs)
cluster_id = np.argmax(probs)

# 4. Select Model
if max_prob < 0.6:
    model = YOLO("general_model.pt")
else:
    # Logic handling specific cluster routing
    if cluster_id == 1: model = YOLO("cluster_0_model.pt")
    elif cluster_id == 2: model = YOLO("cluster_3_model.pt")
    else: model = YOLO(f"cluster_{cluster_id}_model.pt")

# 5. Detect
results = model(input_image)
