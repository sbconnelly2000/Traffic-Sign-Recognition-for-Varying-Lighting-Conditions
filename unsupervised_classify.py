import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import skfuzzy as fuzz
import shutil
from pathlib import Path
import joblib





# ================= CONFIGURATION =================
c = 4  # Number of initial clusters
threshold = 0.60  # Probability threshold for ambiguity
ambiguous_id = c  # If c=4 (indices 0-3), ambiguous becomes 4

BINS = (8, 8, 8) 
n_components_pca = 50 

# Paths
src_img_pattern = "car/all_images_labels/images/*.jpg"
src_lbl_dir = Path("car/all_images_labels/labels")
dest_base_dir = Path(".") # Will create folders in current directory
# =================================================

# 1. LOAD DATA
# We sort the paths to ensure the index matches the file later
image_paths = sorted(glob.glob(src_img_pattern))

print(f"Found {len(image_paths)} images.")

data = []

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        continue
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Calculate Histogram
    hist = cv2.calcHist([hsv], [0, 1, 2], None, BINS, [0, 180, 0, 256, 0, 256])
    
    # Normalize
    cv2.normalize(hist, hist)
    
    data.append(hist.flatten())

# 2. PCA REDUCTION
X = np.array(data)
print(f"Data shape before PCA: {X.shape}")

pca = PCA(n_components=n_components_pca, random_state=42)
X_pca = pca.fit_transform(X)

# Skfuzzy expects (n_features, n_samples), so we transpose
X_fuzzy = X_pca.T 

# 3. RUN FUZZY C-MEANS
print("Running Fuzzy C-Means...")
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X_fuzzy, 
    c=c,  
    m=1.1, 
    error=0.005, 
    maxiter=1000, 
    init=None
)

# 4. APPLY THRESHOLD LOGIC
# Get the standard "hard" labels (0, 1, 2, 3)
# ... after cmeans runs ...

max_probs = np.max(u, axis=0)

print(f"Highest confidence found in dataset: {np.max(max_probs):.4f}")
print(f"Average confidence in dataset: {np.mean(max_probs):.4f}")
print(f"Lowest confidence in dataset: {np.min(max_probs):.4f}")

# ... then your threshold logic ...
cluster_labels = np.argmax(u, axis=0)

# Get the max probability for each point
max_probs = np.max(u, axis=0)

# Assign the new "Ambiguous" ID where confidence is low
cluster_labels[max_probs < threshold] = ambiguous_id

# Print distribution
unique, counts = np.unique(cluster_labels, return_counts=True)
print("Cluster distribution:", dict(zip(unique, counts)))

# 5. VISUALIZATION
# We plot using X_pca (2D projection) rather than X (raw histograms) for better visuals
colors = ['r', 'g', 'b', 'c', 'k'] # Colors for 0, 1, 2, 3, Ambiguous
labels_names = [f'Cluster {i}' for i in range(c)] + ['Ambiguous']

plt.figure(figsize=(10, 8))
for i in range(c + 1): # +1 to include the ambiguous group
    # Find points belonging to this cluster
    mask = (cluster_labels == i)
    
    if np.any(mask):
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=colors[i % len(colors)], 
                   label=labels_names[i],
                   alpha=0.6)

plt.title(f"Fuzzy Clusters (Threshold {threshold*100}%)")
plt.legend()
plt.show()

# 6. FILE COPYING / SORTING
print("Starting file copy operation...")

for i, label in enumerate(cluster_labels):
    # Determine folder name
    if label == ambiguous_id:
        folder_name = "cluster_ambiguous"
    else:
        folder_name = f"cluster_{label}"
        
    # Get filenames
    full_img_path = Path(image_paths[i])
    img_filename = full_img_path.name
    
    # Determine label filename (replace .jpg with .txt)
    lbl_filename = full_img_path.stem + ".txt"
    src_lbl_path = src_lbl_dir / lbl_filename
    
    # Create Destination Paths
    dst_img_folder = dest_base_dir / folder_name / "images"
    dst_lbl_folder = dest_base_dir / folder_name / "labels"
    
    dst_img_folder.mkdir(parents=True, exist_ok=True)
    dst_lbl_folder.mkdir(parents=True, exist_ok=True)
    
    # Perform Copy
    try:
        # Copy Image
        shutil.copy(full_img_path, dst_img_folder / img_filename)
        
        # Copy Label (if it exists)
        if src_lbl_path.exists():
            shutil.copy(src_lbl_path, dst_lbl_folder / lbl_filename)
        else:
            # Optional: Uncomment to see missing labels
            # print(f"Warning: Label missing for {img_filename}")
            pass
            
    except Exception as e:
        print(f"Error processing {img_filename}: {e}")

print("Processing Complete. Check your directories.")

np.save('fcm_centers.npy', cntr)


joblib.dump(pca, 'pca_model.pkl')
