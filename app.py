from flask import Flask, render_template, request, redirect, url_for, send_from_directory, render_template_string
import os
from paddleocr import PaddleOCR
import re
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
# Initialize the OCR model for English
ocr = PaddleOCR(lang='en')
#yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# Function to extract text from an image
def extract_text_from_image(image_path):
    result = ocr.ocr(image_path, cls=True)
    extracted_text = ' '.join([line[1][0] for line in result[0]])
    return extracted_text

# Load the fruit classification model
model = load_model('DenseNet20_model.h5')  # Ensure the path to your model is correct

# Class names for predictions
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

# Preprocess image for classification
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(128, 128))  # Resize to (128, 128)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize image
    return img_array

# Function to extract expiry dates from text
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

# Function to validate expiry date
from datetime import datetime

def is_expired(expiry_date_str):
    if expiry_date_str is None:
        return "Invalid or Missing Expiry Date"

    date_formats = ['%d/%m/%Y', '%Y-%m-%d', '%d-%b-%Y', '%d %b %Y', '%d/%m/%y']  # Add more formats if needed
    for date_format in date_formats:
        try:
            expiry_date = datetime.strptime(expiry_date_str, date_format)
            if expiry_date > datetime.now():
                return False
        except ValueError:
            continue

    return True


def perform_object_detection(image_path):
    results = yolo_model(image_path)
    img = cv2.imread(image_path)
    results.render()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str, len(results.xyxy[0])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Fruit Classifier and OCR</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f0f0f0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background-image: url('https://imgs.search.brave.com/ILIvqq-cAaqUK8fC1qWjtiFifndB-fdH2ZoySt8X2vk/rs:fit:500:0:0:0/g:ce/aHR0cHM6Ly9wbHVz/cG5nLmNvbS9pbWct/cG5nL2xvZ28tZmxp/cGthcnQtcG5nLXJl/Y2VudC1tZW50aW9u/cy02MDAuanBn');
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }
            .container {
                background-color: transparent;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
                text-align: center;
            }
            h1 {
                color: #333;
            }
            .button {
                display: inline-block;
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                text-decoration: none;
                margin: 10px;
            }
            .button:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Welcome to Fruit Classifier and OCR</h1>
            <a class="button" href="/text_extraction">Text Extraction</a>
            <br>
            <a class="button" href="/object_detection">Object Detection</a>
            <br>
            <a class="button" href="/upload">Freshness Prediction</a>
        </div>
    </body>
    </html>
    '''
@app.route('/text_extraction', methods=['GET', 'POST'])
def text_extraction():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)

            # Extract text from the uploaded image
            extracted_text = extract_text_from_image(image_path)
            expiry_date = extract_expiry_date(extracted_text)
            expired_status = is_expired(expiry_date)
            return render_template_string('''
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Text Extraction Result</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        background-color: #f0f0f0;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        margin: 0;
                    }
                    .container {
                        background-color: white;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
                        text-align: center;
                    }
                    h1 {
                        color: #333;
                    }
                    .button {
                        display: inline-block;
                        background-color: #4CAF50;
                        color: white;
                        padding: 10px 20px;
                        border: none;
                        border-radius: 5px;
                        cursor: pointer;
                        text-decoration: none;
                        margin: 10px;
                    }
                                           img {
                        max-width: 300px;
                        margin: 20px 0;
                        border: 2px solid #ddd;
                        border-radius: 8px;
                    }
                    a {
                        text-decoration: none;
                        color: #4CAF50;
                    }
                    .button:hover {
                        background-color: #45a049;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Extracted Text</h1>
                    <img src="/uploads/{{ filename }}" alt="Uploaded Image">
                    <p>{{ extracted_text }}</p>
                    <h2>Expiry Date: {{ expiry_date }}</h2>
                    <h2>Is Expired: {{ expired_status }}</h2>
                    <a class="button" href="/text_extraction">Extract Another Image</a>
                    <br>
                    <a class="button" href="/">Home</a>
                </div>
            </body>
            </html>
            ''', extracted_text=extracted_text, expiry_date=expiry_date, filename=image_file.filename, expired_status=expired_status)

    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Text Extraction</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f0f0f0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }
            .container {
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
                text-align: center;
            }
            h1 {
                color: #333;
            }
            input[type="file"] {
                display: block;
                margin: 20px auto;
            }
            .button {
                display: inline-block;
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                text-decoration: none;
                margin: 10px;
            }
            .button:hover {
                background-color: #45a049;
            }
             img {
                        max-width: 300px;
                        margin: 20px 0;
                        border: 2px solid #ddd;
                        border-radius: 8px;
                    }
                    a {
                        text-decoration: none;
                        color: #4CAF50;
                    }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Upload Image for Text Extraction</h1>
            <form method="POST" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/*" required>
                <br>
                <button class="button" type="submit">Extract Text</button>
            </form>
            <a class="button" href="/">Home</a>
        </div>
    </body>
    </html>
    '''

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)

            # Preprocess image and make a prediction
            img_array = preprocess_image(image_path)
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            predicted_label = class_names[predicted_class]

            # Extract confidence score
            confidence_score = predictions[0][predicted_class]

            return render_template_string('''
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Freshness Prediction Result</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        background-color: #f0f0f0;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        margin: 0;
                    }
                    .container {
                        background-color: white;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
                        text-align: center;
                    }
                    h1 {
                        color: #333;
                    }
                    .button {
                        display: inline-block;
                        background-color: #4CAF50;
                        color: white;
                        padding: 10px 20px;
                        border: none;
                        border-radius: 5px;
                        cursor: pointer;
                        text-decoration: none;
                        margin: 10px;
                    }
                    img {
                        max-width: 300px;
                        margin: 20px 0;
                        border: 2px solid #ddd;
                        border-radius: 8px;
                    }
                    a {
                        text-decoration: none;
                        color: #4CAF50;
                    }
                    .button:hover {
                        background-color: #45a049;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Freshness Prediction Result</h1>
                    <img src="/uploads/{{ filename }}" alt="Uploaded Image">
                    <p><strong>Predicted Class:</strong> {{ predicted_label }}</p>
                    <p><strong>Confidence Score:</strong> {{ confidence_score }}</p>
                    <a class="button" href="/upload">Upload Another Image</a>
                    <br>
                    <a class="button" href="/">Home</a>
                </div>
            </body>
            </html>
            ''', predicted_label=predicted_label, confidence_score=confidence_score, filename=image_file.filename)

    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Upload Image for Freshness Detection</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f0f0f0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }
            .container {
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
                text-align: center;
            }
            h1 {
                color: #333;
            }
            .button {
                display: inline-block;
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                text-decoration: none;
                margin: 10px;
            }
            input[type="file"] {
                padding: 10px;
                margin: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Upload Image for Freshness Detection</h1>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/*" required>
                <br><br>
                <button type="submit" class="button">Submit</button>
            </form>
        </div>
    </body>
    </html>
    '''
@app.route('/uploads/<filename>' )
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
