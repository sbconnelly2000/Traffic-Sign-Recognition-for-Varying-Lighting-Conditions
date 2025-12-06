from ultralytics import YOLO

model = YOLO("yolov8n.yaml")

results = model.train(data = 'data.yaml', epochs = 100, patience = 5, mixup = .1)