import cv2
import numpy as np
from tensorflow.keras.models import load_model
import glob
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model = load_model("digit_model.h5")

# Test on a dataset image to see what color format it expects
# We will use .pngs from dataset_split/train/1
img_path = glob.glob("dataset_split/train/1/*.png")[0]
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
print("Dataset image shape:", img.shape)
print("Dataset image pixel values (min, max, mean):", img.min(), img.max(), img.mean())

# The images might be drawn black-on-white.
# Process like my extraction code (threshold to binary_inv to get white-on-black ink)
_, img_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
img_processed = cv2.resize(img_thresh, (28, 28)).astype("float32") / 255.0
pred1 = model.predict(img_processed.reshape(1, 28, 28, 1), verbose=0)

# Process Normal (black on white) but resize directly
img_norm = cv2.resize(img, (28, 28)).astype("float32") / 255.0
pred2 = model.predict(img_norm.reshape(1, 28, 28, 1), verbose=0)

# Inverted 
img_norm_inv = 1.0 - img_norm
pred3 = model.predict(img_norm_inv.reshape(1, 28, 28, 1), verbose=0)

print(f"Prediction on Thresholded INV (White Ink on Black Bg): {np.argmax(pred1)} (prob: {pred1.max():.2f})")
print(f"Prediction on Normal (Raw Image): {np.argmax(pred2)} (prob: {pred2.max():.2f})")
print(f"Prediction on Inverted Normal: {np.argmax(pred3)} (prob: {pred3.max():.2f})")
