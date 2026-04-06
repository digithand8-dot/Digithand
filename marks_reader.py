import cv2
import easyocr
import re

IMAGE_PATH = "marks.jpg"

reader = easyocr.Reader(['en'], gpu=False)

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(5,5),0)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        15,
        3
    )

    return thresh


def read_marks(image_path):

    img = cv2.imread(image_path)
    img = cv2.resize(img,None,fx=2,fy=2)

    processed = preprocess(img)

    results = reader.readtext(processed)

    print("\n--- OCR DETECTIONS ---\n")

    marks = {}

    for (bbox, text, prob) in results:

        print(text, " confidence:", prob)

        text = text.lower()

        # detect absent
        if "ab" in text:
            num = re.findall(r'\d+', text)
            if num:
                marks[int(num[0])] = "Ab"
            continue

        # detect digits
        nums = re.findall(r'\d+', text)

        if len(nums) >= 2:
            student = int(nums[0])
            mark = nums[1]

            marks[student] = mark

    return marks


def print_table(marks):

    print("\nFINAL MARKS TABLE\n")

    if len(marks)==0:
        print("No marks detected")
        return

    for student in sorted(marks.keys()):
        print(f"Student {student} : {marks[student]}")


marks = read_marks(IMAGE_PATH)
print_table(marks)