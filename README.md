# 🚦 Traffic Sign Recognition using YOLOv8

This project uses **YOLOv8 (You Only Look Once)**, a cutting-edge object detection model, to detect and classify **traffic signs** in real-time images or videos.

## 🧠 Overview

The goal is to accurately identify and locate traffic signs such as **Stop**, **Yield**, **Speed Limit**, and **No Entry** in images or live video feeds using YOLOv8, offering fast and reliable detection for autonomous driving or traffic monitoring systems.


## 🚀 Features

* ⚡ Real-time traffic sign detection
* 🧠 Powered by YOLOv8 (Ultralytics)
* 🗂️ Supports custom datasets in YOLO format
* 📷 Inference on images, videos, or webcam streams
* 📊 Visualizes bounding boxes, class labels, and confidence scores


## 🧰 Requirements

* Python 3.8+
* PyTorch
* OpenCV
* Ultralytics (YOLOv8)

## 📦 Dataset

Use a traffic sign dataset such as:

* [German Traffic Sign Detection Benchmark (GTSDB)](https://benchmark.ini.rub.de/gtsdb_dataset.html)
* Custom datasets in YOLO format


## 📌 Tips

* Use larger YOLO models (`yolov8m.pt`, `yolov8l.pt`) for better accuracy (requires more compute).
* Annotate your own dataset using [Roboflow](https://roboflow.com/) or [LabelImg](https://github.com/tzutalin/labelImg).
* Resize images to 640×640 for best speed/accuracy trade-off.

