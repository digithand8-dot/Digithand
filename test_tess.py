import cv2
import numpy as np

img = cv2.imread('debug_images/05_mark_24_57.jpg', cv2.IMREAD_GRAYSCALE)
sub_contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
sub_boxes = [cv2.boundingRect(sc) for sc in sub_contours]
valid_subs = []
for box in sub_boxes:
    sx, sy, sw, sh = box
    if sw*sh > 30 and sh > 12 and sw < sh * 2.5:
        valid_subs.append(box)
valid_subs.sort(key=lambda b: b[0])
merged_boxes = []
for box in valid_subs:
    if not merged_boxes:
        merged_boxes.append(box)
    else:
        last = merged_boxes[-1]
        if box[0] <= last[0] + last[2] + 4:
            x_min = min(last[0], box[0])
            x_max = max(last[0]+last[2], box[0]+box[2])
            y_min = min(last[1], box[1])
            y_max = max(last[1]+last[3], box[1]+box[3])
            merged_boxes[-1] = (x_min, y_min, x_max - x_min, y_max - y_min)
        else:
            merged_boxes.append(box)

print("Original Boxes:", len(sub_boxes))
print("Valid Boxes:", len(valid_subs))
print("Merged Boxes:", len(merged_boxes))
