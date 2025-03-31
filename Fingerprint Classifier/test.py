import cv2
import numpy as np

# Load image and mask
image = cv2.imread('/Users/jin/Documents/GitHub/AI-Project/Fingerprint Classifier/test_images/imploding whorl/imploding_whorl_1.png')  # or .png, .tif etc.

mask = np.zeros(image.shape[:2], dtype=np.uint8)

# Define circle center and radius
center = (image.shape[1] // 2, image.shape[0] // 2)  # center of image
axes = (100, 50)  # (major_axis_half_length, minor_axis_half_length)

# Ensure mask is binary (0 or 255)
cv2.ellipse(mask, center, axes, angle=0, startAngle=0, endAngle=270, color=255, thickness=-1)

# Apply blur to the entire image
blurred = cv2.GaussianBlur(image, (11, 11), 0)

# Use mask to blend blurred and original image
masked_blur = np.where(mask[:, :, None] == 255, blurred, image)

# Save or show result
cv2.imwrite('blurred_result.jpg', masked_blur)