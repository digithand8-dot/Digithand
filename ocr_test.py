import cv2
import pytesseract
import pandas as pd

# Path to Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load image
img = cv2.imread("marks.jpg")

# Resize for better OCR
img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Remove noise
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Threshold
thresh = cv2.adaptiveThreshold(
    blur,255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11,2
)

# Detect contours (possible rows)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rows = []

for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)

    # Ignore small boxes
    if h > 40 and w > 100:
        roi = img[y:y+h, x:x+w]

        text = pytesseract.image_to_string(roi, config='--psm 6')

        rows.append(text.strip())

# Sort rows
rows = sorted(rows)

print("\nExtracted Rows:\n")

for r in rows:
    print(r)

# Convert OCR text into table
data = []

for r in rows:
    parts = r.split()

    if len(parts) >= 2:
        roll = parts[0]
        marks = parts[-1]

        data.append([roll, marks])

# Create table
df = pd.DataFrame(data, columns=["Roll No", "Marks"])

print("\nTable Format:\n")
print(df)