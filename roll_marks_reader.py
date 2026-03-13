import cv2
import pytesseract
import re

# Path for Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load image
img = cv2.imread("marks.jpg")

# Resize for better OCR
img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur
gray = cv2.GaussianBlur(gray, (5,5), 0)

# Threshold
_, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

# OCR
config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(thresh, config=config)

print("\nRoll Number - Marks\n")

lines = text.split("\n")

for line in lines:

    line = line.strip()

    match = re.findall(r'\d+|Ab|AB', line)

    if len(match) >= 2:
        print(match[0], "-", match[1])