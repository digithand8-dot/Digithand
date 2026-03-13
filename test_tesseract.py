import cv2
import pytesseract

img = cv2.imread('debug_images/01_thresh.jpg', cv2.IMREAD_GRAYSCALE)
img_inv = cv2.bitwise_not(img)
text = pytesseract.image_to_string(img_inv, config='--psm 6').strip()
print("Recognized text length:", len(text))
print("First 100 chars:", text[:100])
