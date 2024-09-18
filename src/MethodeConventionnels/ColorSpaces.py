import cv2 as cv
from skimage.color import rgb2hsv
import numpy as np


def HSV(image):
    
    # Check if image shape is as expected
    if len(image.shape) != 3 or image.shape[2] != 3:
        print("Error: Input image is not in RGB format.")
        return None
    else:
        # Convert RGB image to HSV
        hsv_img = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        return hsv_img


def YCbCr(image):
    
    # Check if image shape is as expected
    if len(image.shape) != 3 or image.shape[2] != 3:
        print("Error: Input image is not in RGB format.")
        return None
    
    else:
        # Convert RGB image to YCbCr
        YCbCr_img = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
        return YCbCr_img

def HSL(image):
    
    # Check if image shape is as expected
    if len(image.shape) != 3 or image.shape[2] != 3:
        print("Error: Input image is not in RGB format.")
        return None
    
    else:
        # Convert RGB image to HLS
        hls_img = cv.cvtColor(image, cv.COLOR_BGR2HLS)
        return hls_img



def LAB(image):
    # Check if image shape is as expected
    if len(image.shape) != 3 or image.shape[2] != 3:
        print("Error: Input image is not in RGB format.")
        return None
    
    else:
        # Convert RGB image to LAB
        lab_img = cv.cvtColor(image, cv.COLOR_BGR2LAB)
        return lab_img



def XYZ(image):
    
    # Check if image shape is as expected
    if len(image.shape) != 3 or image.shape[2] != 3:
        print("Error: Input image is not in RGB format.")
        return None
    
    else:
        # Convert RGB image to XYZ
        XYZ_img = cv.cvtColor(image, cv.COLOR_BGR2XYZ)
        return XYZ_img