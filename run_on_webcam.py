from ultralytics import YOLO
import cv2
import joblib
from sklearn.decomposition import PCA
import numpy as np
import time

cam = cv2.VideoCapture(0)

yolo_model_0 = YOLO("cluster_0/runs/detect/train2/weights/best.pt", task = 'detect')
yolo_model_1 = YOLO("cluster_1/runs/detect/train/weights/best.pt", task = 'detect')
yolo_model_2 = YOLO("cluster_2/runs/detect/train/weights/best.pt", task = 'detect')
yolo_model_3 = YOLO("cluster_3/runs/detect/train/weights/best.pt", task = 'detect')
yolo_general = YOLO("runs/detect/train3/weights/best.pt")

pca_model = joblib.load("pca_model.pkl")

centers = np.load("fcm_centers.npy")

BINS_V = [32]
m = 1.1
power = 2/(m-1)

while True:
    ret, image = cam.read()
    if not ret:
        break
    start_time = time.time()
    image = cv2.resize(image, (480, 480))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [2], None, BINS_V, [0, 256])
    
    X = hist.flatten()
    X = X.reshape(1,-1)

   
    x_pca = pca_model.transform(X)

    
    centers = np.load("fcm_centers.npy")
    
    distances = np.linalg.norm(centers - x_pca, axis=1)

    m = 1.1
    power = 2/(m-1)

    probs = np.zeros(len(centers))

    for i in range(len(centers)):
        probs[i] = np.sum((distances / distances[i]) ** power)
        probs[i] = 1/probs[i]
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
    end_time = time.time()
    print(f'The total inference time is {end_time - start_time}')
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = yolo_model_0.names[cls]

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Pi Camera", image)

        if cv2.waitKey(1):
            break

