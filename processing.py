import cv2
import easyocr
import re
import numpy as np

IMAGE_PATH = "marks.jpg"

# Initialize OCR
reader = easyocr.Reader(['en'], gpu=False)

def preprocess(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(5,5),0)

    thresh = cv2.adaptiveThreshold(
        blur,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,2
    )

    return thresh


def read_marks(image_path):

    img = cv2.imread(image_path)

    img = cv2.resize(img, None, fx=1.8, fy=1.8)

    processed = preprocess(img)

    results = reader.readtext(processed)

    marks = {}

    for (_, text, prob) in results:

        text = text.replace(" ", "")

        match = re.match(r'(\d+)[-:](\d+|Ab|AB|ab)', text)

        if match:

            student = int(match.group(1))
            mark = match.group(2)

            marks[student] = mark

    return marks


def print_table(marks):

    print("\nFINAL MARKS TABLE\n")

    for student in sorted(marks.keys()):

        print(f"Student {student} : {marks[student]}")


if __name__ == "__main__":

    marks = read_marks(IMAGE_PATH)

    print_table(marks)