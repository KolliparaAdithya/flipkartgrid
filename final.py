from flask import Flask, render_template, request, redirect, url_for, send_from_directory, render_template_string
import os
import torch
from paddleocr import PaddleOCR
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
import base64
import re
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize PaddleOCR
ocr = PaddleOCR(lang='en')

# Load .h5 model
fruit_model = load_model('DenseNet20_model.h5')

# Class names for fruit freshness classification
class_names = {
    0: 'Banana_Bad',
    1: 'Banana_Good',
    2: 'Fresh',
    3: 'FreshCarrot',
    4: 'FreshCucumber',
    5: 'FreshMango',
    6: 'FreshTomato',
    7: 'Guava_Bad',
    8: 'Guava_Good',
    9: 'Lime_Bad',
    10: 'Lime_Good',
    11: 'Rotten',
    12: 'RottenCarrot',
    13: 'RottenCucumber',
    14: 'RottenMango',
    15: 'RottenTomato',
    16: 'freshBread',
    17: 'rottenBread'
}

# Helper function: Preprocess image for fruit classification
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Helper function: Extract expiry date from text
def extract_expiry_date(text):
    expiry_date_patterns = [
         r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Expiry Date: 20/07/2O24
    r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-]\d{4}))',  # Expiry Date: 20/07/2024
    r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Expiry Date: 20/07/2O24
    r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Expiry Date: 20 MAY 2O24
    r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*\d{4}))',  # Expiry Date: 20 MAY 2024
    r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Expiry Date: 20 MAY 2O24
    r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Expiry Date: 2024/07/2O24
    r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-]\d{2}))',  # Expiry Date: 2024/07/20
    r'(?:best\s*before\s*[:\-]?\s*.*?(\d{4}))',  # Best Before: 2025
    r'(?:best\s*before\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Best Before: 20/07/2O24
    r'(?:best\s*before\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-]\d{4}))',  # Best Before: 20/07/2024
    r'(?:best\s*before\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Best Before: 20/07/2O24
    r'(?:best\s*before\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Best Before: 20 MAY 2O24
    r'(?:best\s*before\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*\d{4}))',  # Best Before: 20 MAY 2024
    r'(?:best\s*before\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Best Before: 20 MAY 2O24
    r'(?:best\s*before\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Best Before: 2024/07/2O24
    r'(?:best\s*before\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-]\d{2}))',  # Best Before: 2024/07/20
    r'(?:best\s*before\s*[:\-]?\s*.*?(\d{1,2}\d{2}\d{2}))', 
    r'(?:best\s*before\s*[:\-]?\s*(\d{6}))',
    r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{1,2}[A-Za-z]{3,}[0O]\d{2}))',  # Consume Before: 3ODEC2O24
    r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{1,2}[A-Za-z]{3,}\d{2}))',  # Consume Before: 30DEC23
    r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Consume Before: 20/07/2O24
    r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-]\d{4}))',  # Consume Before: 20/07/2024
    r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Consume Before: 20/07/2O24
    r'(?:consume\s*before\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Consume Before: 20 MAY 2O24
    r'(?:consume\s*before\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*\d{4}))',  # Consume Before: 20 MAY 2024
    r'(?:consume\s*before\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Consume Before: 20 MAY 2O24
    r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Consume Before: 2024/07/2O24
    r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-]\d{2}))',  # Consume Before: 2024/07/20
    r'(?:exp\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Exp: 20/07/2O24
    r'(?:exp\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-]\d{4}))',  # Exp: 20/07/2024
    r'(?:exp\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Exp: 20/07/2O24
    r'(?:exp\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Exp: 20 MAY 2O24
    r'(?:exp\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*\d{4}))',  # Exp: 20 MAY 2024
    r'(?:exp\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Exp: 20 MAY 2O24
    r'(?:exp\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Exp: 2024/07/2O24
    r'(?:exp\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-]\d{2}))',  # Exp: 2024/07/20
    r"Exp\.Date\s+(\d{2}[A-Z]{3}\d{4})",
    r'(?:exp\s*\.?\s*date\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Exp. Date: 16 MAR 2O30 (with typo)
    r'(?:exp\s*\.?\s*date\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Exp. Date: 15/12/2O30 (with typo)
    r'(?:exp\s*\.?\s*date\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Exp. Date: 15 MAR 2O30 (with typo)
    r'(?:exp\s*\.?\s*date\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Exp. Date cdsyubfuyef 15 MAR 2O30 (with typo)
    r'(\d{2}[\/\-]\d{2}[\/\-]\d{4})',  # 20/07/2024
    r'(\d{2}[\/\-]\d{2}[\/\-]\d{2})',  # 20/07/24
    r'(\d{2}\s*[A-Za-z]{3,}\s*\d{4})',  # 20 MAY 2024
    r'(\d{2}\s*[A-Za-z]{3,}\s*\d{2})',  # 20 MAY 24
    r'(\d{4}[\/\-]\d{2}[\/\-]\d{2})',  # 2024/07/20
    r'(\d{4}[\/\-]\d{2}[\/\-]\d{2})',  # 2024-07-20
    r'(\d{4}[A-Za-z]{3,}\d{2})',  # 2024MAY20
    r'(\d{2}[A-Za-z]{3,}\d{4})',  # 20MAY2024
    r'(?:DX3\s*[:\-]?\s*(\d{2}\s*\d{2}\s*\d{4}))',
    r'(?:exp\.?\s*date\s*[:\-]?\s*(\d{2}\s*[A-Za-z]{3,}\s*(\d{4}|\d{2})))',
    r'(?:exp\.?\s*date\s*[:\-]?\s*(\d{2}\s*\d{2}\s*\d{4}))',  # Exp. Date: 20 05 2025
    r'(\d{4}[A-Za-z]{3}\d{2})',  # 2025MAY11
    r'(?:best\s*before\s*[:\-]?\s*(\d+)\s*(days?|months?|years?))',  # Best Before: 6 months
    r'(?:best\s*before\s*[:\-]?\s*(three)\s*(months?))',
    r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b\s*\d{4})',
    r'\bUSE BY\s+(\d{1,2}[A-Za-z]{3}\d{4})\b',
    r"Exp\.Date\s*(\d{2}[A-Z]{3}\d{4})",
    r"EXP:\d{4}/\d{2}/\d{4}/\d{1}/[A-Z]"
    ]
    for pattern in expiry_date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None

# Route: Home
@app.route('/')
def index():
    return '''
    <h1>Integrated Application</h1>
    <a href="/object_detection">Object Detection</a><br>
    <a href="/text_extraction">Text Extraction</a><br>
    <a href="/upload">Fruit Freshness Prediction</a>
    '''

# Route: Object Detection
@app.route('/object_detection', methods=['GET', 'POST'])
def object_detection():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)

            # Perform object detection
            results = yolo_model(image_path)
            results.render()

            # Save output image
            output_image = results.ims[0]
            output_path = os.path.join('static', f'detected_{image_file.filename}')
            cv2.imwrite(output_path, output_image)

            return render_template_string('''
            <h1>Object Detection Result</h1>
            <img src="{{ url_for('static', filename='detected_' + filename) }}" alt="Detected Image">
            <br><a href="/">Home</a>
            ''', filename=image_file.filename)
    return '''
    <h1>Upload Image for Object Detection</h1>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" required>
        <button type="submit">Detect</button>
    </form>
    '''

# Route: Text Extraction
@app.route('/text_extraction', methods=['GET', 'POST'])
def text_extraction():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)

            # Extract text
            result = ocr.ocr(image_path)
            text = ' '.join([line[1][0] for line in result[0]])
            expiry_date = extract_expiry_date(text)

            return render_template_string('''
            <h1>Text Extraction Result</h1>
            <p>Extracted Text: {{ text }}</p>
            <p>Expiry Date: {{ expiry_date }}</p>
            <br><a href="/">Home</a>
            ''', text=text, expiry_date=expiry_date)
    return '''
    <h1>Upload Image for Text Extraction</h1>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" required>
        <button type="submit">Extract</button>
    </form>
    '''

# Route: Fruit Freshness Prediction
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)

            # Perform prediction
            img_array = preprocess_image(image_path)
            predictions = fruit_model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            predicted_label = class_names[predicted_class]

            return render_template_string('''
            <h1>Fruit Freshness Prediction</h1>
            <p>Predicted Label: {{ label }}</p>
            <br><a href="/">Home</a>
            ''', label=predicted_label)
    return '''
    <h1>Upload Image for Freshness Prediction</h1>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" required>
        <button type="submit">Predict</button>
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
