from ultralytics import YOLO
import cv2
import joblib
from sklearn.decomposition import PCA
import numpy as np

BINS_V = [32]

image = cv2.imread("street_sign_test.jpg")
image = cv2.resize(image, (416, 416))


yolo_model_0 = YOLO("cluster_0/runs/detect/train2/weights/best.pt", task = 'detect')
yolo_model_1 = YOLO("cluster_1/runs/detect/train/weights/best.pt", task = 'detect')
yolo_model_2 = YOLO("cluster_2/runs/detect/train/weights/best.pt", task = 'detect')
yolo_model_3 = YOLO("cluster_3/runs/detect/train/weights/best.pt", task = 'detect')
yolo_general = YOLO("runs/detect/train3/weights/best.pt")

print('opening pkl')

pca_model = joblib.load("pca_model.pkl")

print("calculate hist")
#test = cv2.imread(image)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    

hist = cv2.calcHist([hsv], [2], None, BINS_V, [0, 256])
   
X = hist.flatten()

X = X.reshape(1,-1)

print("applying pca")
x_pca = pca_model.transform(X)

#Load the cluster centers
centers = np.load("fcm_centers.npy")
#Compute ||x - v_i|| for each cluster
distances = np.linalg.norm(centers - x_pca, axis=1)
#Set constants
m = 1.1
power = 2/(m-1)
#Initialize array of probabilities
probs = np.zeros(len(centers))
#Complete the summation and raise each output to the -1st power
for i in range(len(centers)):
    probs[i] = np.sum((distances / distances[i]) ** power)
    probs[i] = 1/probs[i]
#find the maximum probability and assign the cluster ID
max_prob = np.max(probs)
cluster_id = np.argmax(probs)

if max_prob < .6:
    final_label = "ambiguous"
else:
    final_label = cluster_id

if final_label == "ambiguous":
    print("ambiguous")
    selected_model = yolo_general
elif final_label == 0:
    print("cluster 0")
    selected_model = yolo_model_0
elif final_label == 1:
    print("cluster 1")
    selected_model = yolo_model_0
elif final_label == 2:
    print("cluster 2")
    selected_model = yolo_model_3
elif final_label == 3:
    print(f"cluster 3")
    selected_model = yolo_model_3

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
