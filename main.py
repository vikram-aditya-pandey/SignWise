import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
from ultralytics import YOLO
model = YOLO("yolov8n.yaml")
results = model.train(data="config.yaml", epochs=2)