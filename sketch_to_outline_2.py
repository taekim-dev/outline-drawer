import cv2
import numpy as np
import json

# Load requirements
with open('output_requirement.json', 'r') as f:
    req = json.load(f)

# Parameters from requirements
outline_color = req.get('outline_color', '#000000')
background_color = req.get('background_color', '#FFFFFF')
line_thickness = req.get('line_thickness_px', 5)
anti_aliasing = req.get('anti_aliasing_allowed', False)
closed_shapes_required = req.get('closed_shapes_required', True)
margin_percent = req.get('margin_percent', 10)
image_size = req.get('image_size', [512, 512])
allow_noise = req.get('allow_noise', False)
interior_should_be_empty = req.get('interior_should_be_empty', True)

# 1. Load and resize image
img = cv2.imread('input_sketch.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Median blur to reduce paper texture
blurred = cv2.medianBlur(gray, 5)

# 3. Threshold to pure black/white
thresh = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
)

# 4. Morphological opening with a small kernel to remove noise but preserve lines
kernel_open = np.ones((3, 3), np.uint8)
opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)

# 5. Find all contours (preserve all linework)
contours, hierarchy = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# 6. Filter out only very small contours (noise)
min_area = 30
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# 7. Draw all contours as bold outlines (not filled)
mask = np.zeros_like(thresh)
cv2.drawContours(mask, filtered_contours, -1, 255, thickness=line_thickness)

# 8. Morphological closing to connect and smooth lines
kernel_close = np.ones((line_thickness, line_thickness), np.uint8)
closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

# 9. Dilation to further bolden lines (optional)
dilated = cv2.dilate(closed, kernel_close, iterations=1)

# 10. Center subject and add margin
ys, xs = np.where(dilated > 0)
if len(xs) > 0 and len(ys) > 0:
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    h, w = dilated.shape
    margin = int(min(h, w) * margin_percent / 100)
    crop = dilated[max(0, y_min-margin):min(h, y_max+margin), max(0, x_min-margin):min(w, x_max+margin)]
    # Resize to required size
    result = cv2.resize(crop, tuple(image_size), interpolation=cv2.INTER_NEAREST)
else:
    result = np.zeros(image_size, dtype=np.uint8)

# 11. Invert for black lines on white
result = cv2.bitwise_not(result)

# 12. Save as PNG
cv2.imwrite('output_outline.png', result)
print('Outline image saved as output_outline.png') 