import base64
import io
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2
import logging
from ultralytics import YOLO
from flask_cors import CORS


def load_model():
    try:
        model = YOLO("best.pt")
        model.fuse()
    except:
        print("model not found")
    return model

model = load_model() 

def predict(self, frame):
        results = self.model(frame)
        return  results

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def plot_bboxes(results, source):
    source_copy = source.copy() 
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            box = box.astype(int)
            cv2.rectangle(source_copy, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    return source_copy





app = Flask(__name__)
CORS(app, origins='http://localhost:3000', methods=['GET', 'POST'], allow_headers=['Content-Type', 'Authorization'])

logging.basicConfig(level=logging.ERROR)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

@app.route('/ping')
def ping():
    return jsonify({'message': 'Backend is alive!'})

@app.route('/detect', methods=['POST'])

def detect_objects():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        print("file: ",file)
        print("filename: ",file.filename)

        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and not allowed_file(file.filename):
            return jsonify({'error': 'Unsupported file format'}), 400

        filename = secure_filename(file.filename)
        
        image_path = f'uploads/{filename}'
        file.save(image_path)
        
        source = Image.open(image_path)
        source = np.array(source.convert('RGB')) 
        results = model(source)

        frame = plot_bboxes(results,source)
        print("results",results)
        pil_image = Image.fromarray(frame)

        img_byte_array = io.BytesIO()
        pil_image.save(img_byte_array, format="JPEG")
        img_byte_array = img_byte_array.getvalue()
        
        encoded_image = base64.b64encode(img_byte_array).decode('utf-8')
        
        return jsonify({'image': encoded_image}), 200
    except Exception as e:
        print(f"Error during processing: {e}")
        return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
