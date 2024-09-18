from PIL import Image
import cv2 as cv
import numpy as np

# Function to flip an image
def image_flipping(image, flipCode):
    """
    This function is used to flip an image.
    """
    # Convert PIL Image to OpenCV format
    image_cv = np.array(image)
    flipped_image = cv.flip(image_cv, flipCode)
    # Convert back to PIL Image
    flipped_image_pil = Image.fromarray(flipped_image)
    return flipped_image_pil