import numpy as np
from PIL import Image
import cv2 as cv

def rotate_image(image, angle=None):
    """
    Rotate the input image by the specified angle.

    :param image: Image d'entrée (PIL.Image).
    :param angle: Angle de rotation (en degrés).
    :return: Image rotated (PIL.Image).
    """
    # Assure that the input is a PIL image
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    rotated_image = image.rotate(angle)
    return rotated_image
