import cv2
import numpy as np

# Read the input image in grayscale
img = cv2.imread('input_sketch.png', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur to reduce noise
g_blur = cv2.GaussianBlur(img, (5, 5), 0)

# Use adaptive thresholding to binarize the image
thresh = cv2.adaptiveThreshold(
    g_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
)

# Morphological operations to thicken and smooth lines
kernel = np.ones((3, 3), np.uint8)
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
dilated = cv2.dilate(closed, kernel, iterations=1)

# Invert image so lines are black, background is white
outline = cv2.bitwise_not(dilated)

# Save the result
cv2.imwrite('output_outline.png', outline)

print('Outline image saved as output_outline.png') 