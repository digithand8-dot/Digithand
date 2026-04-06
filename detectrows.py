import cv2

img = cv2.imread("marks.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thresh = cv2.adaptiveThreshold(
    gray,255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11,2
)

# find contours
contours,_ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

for c in contours:

    x,y,w,h = cv2.boundingRect(c)

    if h > 25 and w > 100:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow("rows",img)
cv2.waitKey(0)
cv2.destroyAllWindows()