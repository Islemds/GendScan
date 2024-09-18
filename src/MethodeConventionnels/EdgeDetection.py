# image_processing.py

import cv2
import numpy as np
from PIL import Image

def canny_edge_detection(image, min_val, max_val):
    """
    Apply Canny edge detection on the input image.
    
    Args:
    image (numpy.ndarray): The input image (grayscale or RGB).
    min_val (int): The minimum threshold for edge detection.
    max_val (int): The maximum threshold for edge detection.
    
    Returns:
    numpy.ndarray: The image with detected edges.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return cv2.Canny(gray, min_val, max_val)

def apply_edge_detection(image, min_val=100, max_val=200):
    """
    Wrapper function to apply edge detection using default thresholds.
    
    Args:
    image (PIL.Image): The input image.
    min_val (int): Minimum threshold for Canny.
    max_val (int): Maximum threshold for Canny.
    
    Returns:
    PIL.Image: The image with detected edges.
    """
    # Convert PIL Image to OpenCV format
    image_cv = np.array(image)
    
    # Convert RGB to BGR if the image is in color
    if len(image_cv.shape) == 3 and image_cv.shape[2] == 3:
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    
    # Apply Canny edge detection
    edges = canny_edge_detection(image_cv, min_val, max_val)
    
    # Convert back to PIL Image format
    return Image.fromarray(edges)
