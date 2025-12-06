from ultralytics import YOLO

# Load the model
model = YOLO("runs/train/train/weights/best.pt")

# Run the evaluation
results = model.val(data="data.yaml")