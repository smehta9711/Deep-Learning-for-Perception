import cv2
import numpy as np

# Load image
image = cv2.imread('mask_907.png', cv2.IMREAD_GRAYSCALE)

# Binarize image (assuming object is white on black background)
_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# Perform connected component labeling
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

# Create an output image with labels
output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

# Assign a color to each label
colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=int)
for label in range(1, num_labels):
    mask = (labels == label)
    output_image[mask] = colors[label]

# Show or save the output image
# cv2.imshow('Segmented Image_cc', output_image)
cv2.imwrite('CCm1.png',output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
