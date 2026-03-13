import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TF warnings
from flask import Flask, request, jsonify, redirect, url_for, session, send_from_directory
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
# Change this to a random secret key if you use sessions
app.secret_key = 'your_secret_key' 

# Load the trained model at startup to save time during requests
model = None
try:
    model = load_model("digit_model.h5")
except Exception as e:
    print("Warning: Could not load digit_model.h5")

def preprocess_for_prediction(roi):
    """ Resizes and pads the extracted contour to 28x28 for the CNN. """
    h, w = roi.shape
    if h > w:
        new_h, new_w = 20, int(20 * w / h)
    else:
        new_h, new_w = int(20 * h / w), 20
    if new_h == 0 or new_w == 0:
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    top = (28 - new_h) // 2
    bottom = 28 - new_h - top
    left = (28 - new_w) // 2
    right = 28 - new_w - left
    
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    padded = padded.astype("float32") / 255.0
    return padded.reshape(1, 28, 28, 1)

def read_marks(image_bytes):
    if model is None:
        return {"error": "Model not loaded on server."}

    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    # 1. Grayscale and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    
    cv2.imwrite("debug_images/01_thresh.jpg", thresh)
    
    # 2. Extract Connected Components
    # We dilate horizontally to group "24", "Ab", etc., into single words
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    cv2.imwrite("debug_images/03_cleaned.jpg", cleaned)
    
    # 3. Contour Detection
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    debug_img = img.copy()
    valid_contours = []
    h, w = thresh.shape
    
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if 50 < area < (h*w*0.05) and 10 < ch < 100:
            valid_contours.append((x, y, cw, ch, c))
            
    # Group into 4 columns based on page width
    cols = [[] for _ in range(4)]
    col_width = w / 4.0
    for box in valid_contours:
        x, y, cw, ch, c = box
        cx = x + cw / 2.0
        c_idx = int(cx // col_width)
        if c_idx >= 4: c_idx = 3
        if c_idx < 0: c_idx = 0
        cols[c_idx].append(box)

    results = []
    total_marks = 0
    q_no = 1
    import pytesseract

    for c_idx, col_boxes in enumerate(cols):
        # Sort vertically inside column
        col_boxes.sort(key=lambda b: b[1])
        
        lines = []
        current_line = []
        for box in col_boxes:
            if not current_line:
                current_line.append(box)
            else:
                last_box = current_line[-1]
                # If on same physical line (allow 20px dev)
                if abs(box[1] - last_box[1]) < 20:
                    current_line.append(box)
                else:
                    lines.append(current_line)
                    current_line = [box]
        if current_line:
            lines.append(current_line)
            
        for line in lines:
            if len(line) < 2:
                continue
                
            line.sort(key=lambda b: b[0]) # Left to right
            
            # The right-most block is the mark
            mark_box = line[-1]
            x, y, cw, ch, c = mark_box
            cv2.rectangle(debug_img, (x, y), (x+cw, y+ch), (0, 255, 0), 2)
            
            # Crop the mark from the original clean threshold map
            digit_roi = thresh[y:y+ch, x:x+cw]
            
            # 1. OCR attempt for 'Ab' Check
            img_inv = cv2.bitwise_not(digit_roi)
            padded = cv2.copyMakeBorder(img_inv, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
            config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789Ab'
            text_pred = pytesseract.image_to_string(padded, config=config).strip()
            
            mark_val = 0
            text_str = ""
            
            if "a" in text_pred.lower() or "b" in text_pred.lower():
                text_str = "Ab"
                mark_val = 0
            else:
                # 2. Fallback to CNN for accurate numbers
                # Find exact, non-merged digits inside this block
                sub_contours, _ = cv2.findContours(digit_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                sub_boxes = [cv2.boundingRect(sc) for sc in sub_contours]
                
                # Filter out pure noise, dots, or horizontal dashes
                valid_subs = []
                for box in sub_boxes:
                    sx, sy, sw, sh = box
                    # A valid digit should have some area and be taller than 12 pixels
                    # We also ignore boxes that are extremely wide compared to their height (like dashes)
                    if sw*sh > 30 and sh > 12 and sw < sh * 2.5:
                        valid_subs.append(box)
                        
                valid_subs.sort(key=lambda b: b[0]) # Left to right
                
                # Merge sub-boxes that overlap horizontally or are very close
                # This prevents digits like '4' from being split into two CNN predictions
                merged_boxes = []
                for box in valid_subs:
                    if not merged_boxes:
                        merged_boxes.append(box)
                    else:
                        last = merged_boxes[-1]
                        # If x-coordinates overalap or gap is less than 4 pixels
                        if box[0] <= last[0] + last[2] + 4:
                            x_min = min(last[0], box[0])
                            x_max = max(last[0]+last[2], box[0]+box[2])
                            y_min = min(last[1], box[1])
                            y_max = max(last[1]+last[3], box[1]+box[3])
                            merged_boxes[-1] = (x_min, y_min, x_max - x_min, y_max - y_min)
                        else:
                            merged_boxes.append(box)
                
                if not merged_boxes:
                    text_str = "0"
                else:
                    detected_digits = []
                    for sx, sy, sw, sh in merged_boxes:
                        single_digit = digit_roi[sy:sy+sh, sx:sx+sw]
                        processed = preprocess_for_prediction(single_digit)
                        pred = model(processed, training=False)
                        detected_digits.append(str(int(np.argmax(pred))))
                        
                    text_str = "".join(detected_digits)
                    try:
                        mark_val = int(text_str)
                    except ValueError:
                        mark_val = 0
            
            results.append({
                "q_no": q_no,
                "marks": text_str
            })
            total_marks += mark_val
            
            # Use fixed increment across columns
            # Column 1 starts 1, Column 2 starts ~20, etc. 
            # Actually just trust q_no counter based on loops!
            q_no += 1
            
    cv2.imwrite("debug_images/06_target_marks.jpg", debug_img)
        
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