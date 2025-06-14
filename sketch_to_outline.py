import cv2
import numpy as np

# 1. Load image and convert to grayscale
img = cv2.imread('input_sketch.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Apply median blur to reduce paper texture
blurred = cv2.medianBlur(gray, 5)

# 3. Apply adaptive thresholding to isolate dark lines
thresh = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
)

# 4. Remove small noise using contour area filtering
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros_like(thresh)
min_area = 100  # Minimum area to keep
for cnt in contours:
    if cv2.contourArea(cnt) > min_area:
        cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

# 5. Apply morphological dilation and closing to thicken and join lines
kernel = np.ones((5, 5), np.uint8)
dilated = cv2.dilate(mask, kernel, iterations=1)
closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

# 6. Invert the image (white background, black lines)
result = cv2.bitwise_not(closed)

# 7. Save/export result
cv2.imwrite('output_outline.png', result)

print('Outline image saved as output_outline.png') 