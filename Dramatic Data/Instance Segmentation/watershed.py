import cv2
import numpy as np

# Load image
image = cv2.imread('mask_948.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Preprocessing
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, binary_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Distance transform
distance_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(distance_transform, 0.7 * distance_transform.max(), 255, cv2.THRESH_BINARY)
sure_fg = np.uint8(sure_fg)

# Unknown region
sure_bg = cv2.dilate(binary_image, np.ones((3, 3), np.uint8))
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labeling
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# Apply watershed
markers = cv2.watershed(image, markers)
image[markers == -1] = [0, 0, 255]  # Boundary in red

# Show or save result
cv2.imwrite('refined_water.png', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
