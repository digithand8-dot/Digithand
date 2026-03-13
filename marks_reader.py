import cv2
import easyocr
import re

# Load image
img = cv2.imread("marks.jpg")

# Resize for better recognition
img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Increase contrast
thresh = cv2.adaptiveThreshold(
    gray,255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,2
)

# Initialize OCR
reader = easyocr.Reader(['en'])

# Detect text
results = reader.readtext(thresh)

print("\nDetected Marks:\n")

for (_, text, prob) in results:

    # Remove spaces
    text = text.replace(" ", "")

    # Match patterns like 2-36 or 7-Ab
    match = re.match(r'(\d+)[-:](\w+)', text)

    if match:
        question = match.group(1)
        mark = match.group(2)

        print("Question", question, ":", mark)