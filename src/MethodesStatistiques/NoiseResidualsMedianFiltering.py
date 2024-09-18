import numpy as np
from PIL import Image
import cv2

# Median Filter and Residuals Function
def median_filter_and_residuals(img, kernel_size=3, amplification_factor=30):
    # Apply median filter
    filtered_img = cv2.medianBlur(img, kernel_size)
    
    # Calculate residuals
    residuals_img = cv2.absdiff(img, filtered_img)
    
    # Amplify the residuals
    amplified_residuals = residuals_img * amplification_factor
    
    # Normalize amplified residuals to 0-255 for display
    amplified_residuals_normalized = np.clip(amplified_residuals, 0, 255).astype(np.uint8)
    
    return amplified_residuals_normalized