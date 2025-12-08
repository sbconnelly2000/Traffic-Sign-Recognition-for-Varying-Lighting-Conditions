from ultralytics import YOLO
import cv2
import joblib
from sklearn.decomposition import PCA
import numpy as np

BINS = (8, 8, 8) 
image = cv2.imread("IMG_3560.jpg")
#image = cv2.resize(image, (640, 640))

print("loading models")
yolo_model_1 = YOLO("cluster_1/runs/detect/train2/weights/best.pt", task = 'detect')
yolo_model_2 = YOLO("cluster_2/runs/detect/train/weights/best.pt", task = 'detect')
yolo_model_0_3 = YOLO("cluster_3/runs/detect/train4/weights/best.pt", task = 'detect')
yolo_general = YOLO("runs/detect/train3/weights/best.pt")

print('opening pkl')

pca_model = joblib.load("pca_model.pkl")

print("calculate hist")
#test = cv2.imread(image)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hist = cv2.calcHist([hsv], [0, 1, 2], None, BINS, [0, 180, 0, 256, 0, 256])
cv2.normalize(hist, hist)
X = hist.flatten()

X = X.reshape(1,-1)

print("applying pca")
x_pca = pca_model.transform(X)

centers = np.load("fcm_centers.npy")
distances = np.linalg.norm(centers - x_pca, axis=1)
distances = np.maximum(distances, 1e-10) 

power = 2.0 / .1
probs = np.zeros(len(centers))

for i in range(len(centers)):
    probs[i] = 1.0 / np.sum((distances[i] / distances) ** power)

max_prob = np.max(probs)
cluster_id = np.argmax(probs)


if max_prob < .5:
    final_label = "ambiguous"
else:
    final_label = cluster_id

if final_label == "ambiguous":
    print("ambiguous")
    selected_model = yolo_general
elif final_label == 1:
    print("cluster 1")
    selected_model = yolo_model_1
elif final_label == 2:
    print("cluster 2")
    selected_model = yolo_model_2
elif final_label == 0 or final_label == 3:
    print(f"cluster 0 or 3")
    selected_model = yolo_model_0_3

results = selected_model(image)

image_results = results[0].plot()
        
cv2.imshow("YOLOv8 Image Detection", image_results)
cv2.waitKey(0)
cv2.destroyAllWindows()

#compare to normal model
results = yolo_general(image)

image_results = results[0].plot()
        
cv2.imshow("YOLOv8 Image Detection", image_results)
cv2.waitKey(0)
cv2.destroyAllWindows()
