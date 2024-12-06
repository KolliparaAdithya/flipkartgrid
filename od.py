from flask import Flask, render_template, request, redirect, url_for, send_from_directory, render_template_string
import os
import torch
from matplotlib import pyplot as plt
import cv2
from io import BytesIO
from PIL import Image
import base64

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

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
        <title>Object Detection, Classification, and OCR</title>
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
            .button:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Welcome to Object Detection, Classification, and OCR</h1>
            <a class="button" href="/text_extraction">Text Extraction</a>
            <br>
            <a class="button" href="/upload">Freshness Prediction</a>
            <br>
            <a class="button" href="/object_detection">Object Detection</a>
        </div>
    </body>
    </html>
    '''

@app.route('/object_detection', methods=['GET', 'POST'])
def object_detection():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)

            # Perform object detection using YOLOv5
            results = model(image_path)

            # Render results on the image
            results.render()

            # Extract the image with bounding boxes
            output_image_with_boxes = results.ims[0]  # Use 'ims' instead of 'imgs'

            # Specify the new output directory
            output_folder = 'output_images/'
            os.makedirs(output_folder, exist_ok=True)

            # Save the image with bounding boxes
            output_image_path = os.path.join(output_folder, f'detected_{image_file.filename}')
            cv2.imwrite(output_image_path, output_image_with_boxes)

            # Convert image from BGR to RGB for Matplotlib
            img_rgb = cv2.cvtColor(output_image_with_boxes, cv2.COLOR_BGR2RGB)

            # Save the image with bounding boxes as a BytesIO object
            pil_img = Image.fromarray(img_rgb)
            buffer = BytesIO()
            pil_img.save(buffer, format="JPEG")
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

            detected_objects = []
            for i, (det) in enumerate(results.pred[0]):
                box = det[:4]
                conf = det[4]
                cls = int(det[5])  # Class index
                detected_objects.append(f'Object {i}: {results.names[cls]} (Confidence Score = {conf:.2f})')

            return render_template_string('''
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Object Detection Result</title>
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
                    .button:hover {
                        background-color: #45a049;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Object Detection Result</h1>
                    <img src="data:image/jpeg;base64,{{ img_str }}" alt="Object Detection Result">
                    <p>Number of objects detected: {{ num_objects }}</p>
                    <h2>Detected Objects and Confidence Scores:</h2>
                    <ul>
                        {% for obj in detected_objects %}
                            <li>{{ obj }}</li>
                        {% endfor %}
                    </ul>
                    <a class="button" href="/object_detection">Detect Another Image</a>
                    <br>
                    <a class="button" href="/">Home</a>
                </div>
            </body>
            </html>
            ''', img_str=img_str, num_objects=len(results.pred[0]), detected_objects=detected_objects)

    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Object Detection</title>
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
            }
            .button:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Upload Image for Object Detection</h1>
            <form method="POST" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/*" required>
                <button type="submit" class="button">Detect Objects</button>
                <br>
            </form>
            <br>
            <a class="button" href="/">Home</a>
        </div>
    </body>
    </html>
    '''


if __name__ == '__main__':
    app.run(debug=True, port=5001)
