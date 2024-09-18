import cv2
import numpy as np
from PIL import Image

def gaussian_filter(image, kernel_size=3, sigma=1):
    """
    Applique un filtre gaussien à l'image.
    
    :param image: Image d'entrée (PIL.Image).
    :param kernel_size: Taille du noyau (doit être impair).
    :param sigma: Écart-type de la distribution gaussienne.
    :return: Image filtrée (PIL.Image).
    """
    image_cv = np.array(image)
    filtered_image_cv = cv2.GaussianBlur(image_cv, (kernel_size, kernel_size), sigma)
    return Image.fromarray(filtered_image_cv)

def median_filter(image, kernel_size=3):
    """
    Applique un filtre médian à l'image.
    
    :param image: Image d'entrée (PIL.Image).
    :param kernel_size: Taille du noyau (doit être impair).
    :return: Image filtrée (PIL.Image).
    """
    image_cv = np.array(image)
    filtered_image_cv = cv2.medianBlur(image_cv, kernel_size)
    return Image.fromarray(filtered_image_cv)
