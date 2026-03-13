from flask import Flask, request, jsonify, redirect, url_for, session, send_from_directory
import cv2
import pytesseract
import numpy as np
import re

app = Flask(__name__)
# Change this to a random secret key if you use sessions
app.secret_key = 'your_secret_key' 

# Tesseract path - commented out so it uses system default on Linux
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def read_marks(image):

    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)

    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5,5), 0)

    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    config = r'--oem 3 --psm 6'

    text = pytesseract.image_to_string(thresh, config=config)

    results = []

    lines = text.split("\n")

    for line in lines:

        line = line.strip()

        match = re.findall(r'\d+|Ab|AB', line)

        if len(match) >= 2:

            results.append({
                "roll": match[0],
                "marks": match[1]
            })

    return results


@app.route('/scan', methods=['POST'])
def scan():

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files['image']

    image = file.read()

    data = read_marks(image)

    return jsonify(data)


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)


if __name__ == "__main__":
    app.run(debug=True)