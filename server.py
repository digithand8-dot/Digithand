from flask import Flask, request, jsonify, redirect, url_for, session, send_from_directory
import cv2
import pytesseract
import numpy as np
import re
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a random secret key

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def get_db():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )''')

init_db()


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

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify({"error": "All fields are required"}), 400

    hashed_password = generate_password_hash(password)

    with get_db() as conn:
        try:
            conn.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                         (username, email, hashed_password))
            return jsonify({"message": "Account created successfully"}), 201
        except sqlite3.IntegrityError:
            return jsonify({"error": "Username or email already exists"}), 409

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    with get_db() as conn:
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()

    if user and check_password_hash(user['password'], password):
        session['user_id'] = user['id']
        return jsonify({"message": "Login successful"}), 200
    else:
        return jsonify({"error": "Invalid username or password"}), 401

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)


if __name__ == "__main__":
    app.run(debug=True)