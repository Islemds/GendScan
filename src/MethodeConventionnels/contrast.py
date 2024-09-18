import cv2 as cv
import numpy as np
from PIL import Image

def clahe_contrast(image):
    """
    Apply CLAHE to the image for contrast enhancement.
    """
    # Convert PIL Image to OpenCV format
    image_cv = np.array(image)

    if len(image_cv.shape) == 3 and image_cv.shape[2] >= 3:
        if image_cv.shape[2] == 4:
            image_cv = cv.cvtColor(image_cv, cv.COLOR_RGBA2RGB)
        else:
            image_cv = cv.cvtColor(image_cv, cv.COLOR_BGR2RGB)

    if len(image_cv.shape) == 3:
        lab_img = cv.cvtColor(image_cv, cv.COLOR_RGB2Lab)
        l, a, b = cv.split(lab_img)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(l)
        updated_lab_img = cv.merge((clahe_img, a, b))
        CLAHE_img = cv.cvtColor(updated_lab_img, cv.COLOR_Lab2RGB)
        CLAHE_img = Image.fromarray(CLAHE_img)
        
    return CLAHE_img
