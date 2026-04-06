import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import easyocr
import re

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Initialize EasyOCR once at startup
print("Loading EasyOCR model...")
reader = easyocr.Reader(['en'], gpu=False)
print("EasyOCR ready.")

os.makedirs("debug_images", exist_ok=True)


def preprocess_image(img):
    """
    Enhance image for OCR:
    - Upscale 2x for better recognition
    - Convert to grayscale
    - Apply CLAHE for contrast normalisation
    - Adaptive threshold to clean up ink
    """
    # Upscale for cleaner text
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Adaptive thresholding to get clean binary image
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15, 4
    )
    return img, thresh


def group_detections_into_rows(detections, row_tolerance=25):
    """
    Groups OCR bounding boxes into horizontal rows by their Y centre.
    Returns a list of rows, each row is a list of detections sorted left-to-right.
    """
    if not detections:
        return []

    # Sort all detections by y-centre
    detections = sorted(detections, key=lambda d: (d[0][0][1] + d[0][2][1]) / 2)

    rows = []
    current_row = [detections[0]]
    current_y = (detections[0][0][0][1] + detections[0][0][2][1]) / 2

    for det in detections[1:]:
        det_y = (det[0][0][1] + det[0][2][1]) / 2
        if abs(det_y - current_y) <= row_tolerance:
            current_row.append(det)
        else:
            rows.append(sorted(current_row, key=lambda d: d[0][0][0]))  # sort by x
            current_row = [det]
            current_y = det_y

    if current_row:
        rows.append(sorted(current_row, key=lambda d: d[0][0][0]))

    return rows


def extract_mark_from_row(row_texts):
    """
    Given a list of text tokens from a single row,
    determine the mark value. The mark is typically the last number
    or 'Ab' token in the row.
    Returns (mark_str, mark_int)
    """
    combined = " ".join(row_texts).strip()

    # Check for Absent first
    if re.search(r'\bab\b', combined.lower()):
        return "Ab", 0

    # Find all numeric tokens in the row
    numbers = re.findall(r'\d+', combined)
    if not numbers:
        return None, None

    # The marks column should be the LAST number in the row.
    mark_str = numbers[-1]
    try:
        mark_val = int(mark_str)
        # Sanity check: marks are usually 0-100
        if 0 <= mark_val <= 100:
            return mark_str, mark_val
    except ValueError:
        pass

    return None, None


def read_marks(image_bytes):
    """
    Main entry point: takes image bytes, runs EasyOCR, groups into rows,
    extracts marks and returns structured result.
    """
    img_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Could not decode image"}

    # Preprocess
    upscaled, thresh = preprocess_image(img)
    cv2.imwrite("debug_images/01_thresh.jpg", thresh)

    # Run EasyOCR on the preprocessed image
    # Use raw image for better results (EasyOCR handles preprocessing internally)
    detections = reader.readtext(upscaled, detail=1, paragraph=False)

    # Save debug image with all detected boxes
    debug_img = upscaled.copy()
    for (bbox, text, conf) in detections:
        pts = np.array(bbox, np.int32)
        cv2.polylines(debug_img, [pts], True, (0, 200, 255), 2)
        cv2.putText(debug_img, f"{text}({conf:.2f})",
                    (int(bbox[0][0]), int(bbox[0][1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    cv2.imwrite("debug_images/02_ocr_boxes.jpg", debug_img)

    # Filter low confidence
    detections = [(bbox, text, conf) for (bbox, text, conf) in detections if conf > 0.25]

    if not detections:
        return {"questions": [], "total_marks": 0, "message": "No text detected"}

    # Group into horizontal rows
    rows = group_detections_into_rows(detections, row_tolerance=30)

    results = []
    total_marks = 0
    roll_no = 1

    for row in rows:
        row_texts = [det[1] for det in row]
        mark_str, mark_val = extract_mark_from_row(row_texts)

        if mark_str is not None:
            results.append({
                "roll_no": roll_no,
                "marks": mark_str
            })
            total_marks += mark_val
            roll_no += 1

    # Draw final results
    result_img = upscaled.copy()
    for row in rows:
        for (bbox, text, conf) in row:
            pts = np.array(bbox, np.int32)
            cv2.polylines(result_img, [pts], True, (0, 255, 0), 2)
    cv2.imwrite("debug_images/03_final_marks.jpg", result_img)

    return {
        "questions": results,
        "total_marks": total_marks
    }


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