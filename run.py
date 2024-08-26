import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

class ObjectDetection:

    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        self.model = self.load_model()

    def load_model(self):
        model = YOLO("C:/Users/adity/Downloads/archive/runs/detect/train12(70)/weights/best.pt")  # loading a pretrained model
        model.fuse()
        return model

    def predict(self, frame):
        results = self.model(frame)
        return results

    def plot_bboxes(self, results, frame):
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()

            for box in boxes:
                # Convert the coordinates to integers
                box = box.astype(int)

                # Draw rectangle on the frame
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        return frame

# Example: Create an instance with capture index 0
obj_detection = ObjectDetection(capture_index=0)

# Example: Capture frames from the camera
cap = cv2.VideoCapture(0)  # Use the camera with index 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the frame
    results = obj_detection.predict(frame)

    # Plot bounding boxes on the frame
    frame_with_bboxes = obj_detection.plot_bboxes(results, frame)

    # Display the frame with bounding boxes
    cv2.imshow("Object Detection", frame_with_bboxes)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
