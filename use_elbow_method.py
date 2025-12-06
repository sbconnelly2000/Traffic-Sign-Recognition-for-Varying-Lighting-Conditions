import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import skfuzzy as fuzz 


BINS = (8, 8, 8) 
n_components_pca = 50 

image_paths = glob.glob("car/all_images_labels/images/*.jpg")

print(f"Found {len(image_paths)} images.")

data = []


for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        continue
    
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    
    hist = cv2.calcHist([hsv], [0, 1, 2], None, BINS, [0, 180, 0, 256, 0, 256])
    
    
    cv2.normalize(hist, hist)
    

    data.append(hist.flatten())

X = np.array(data)


pca = PCA(n_components=n_components_pca, random_state=42)
X_pca = pca.fit_transform(X)



X_fuzzy = X_pca.T 
fpcs = [] 


n_clusters_range = range(2, 21) 

print("Running Fuzzy C-Means loop...")
for ncenters in n_clusters_range:
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X_fuzzy, 
        c=ncenters,  
        m=2, 
        error=0.005, 
        maxiter=1000, 
        init=None
    )
    fpcs.append(fpc)
    print(f"Cluster {ncenters} FPC: {fpc:.4f}")

fig, ax = plt.subplots()
ax.plot(n_clusters_range, fpcs, 'bx-')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Fuzzy Partition Coefficient (FPC)')
ax.set_title('Elbow Methods)')
plt.grid(True)
plt.show()