import cv2
import easyocr
import re

# Load image
img = cv2.imread("marks.jpg")

# Resize
img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold
thresh = cv2.adaptiveThreshold(
    gray,255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,2
)

# OCR
reader = easyocr.Reader(['en'])
results = reader.readtext(thresh)

print("\nDetected Marks:\n")

for (_, text, prob) in results:
    numbers = re.findall(r'\d+', text)
    for n in numbers:
        print(n)