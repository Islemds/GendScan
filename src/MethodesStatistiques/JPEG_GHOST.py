import cv2 as cv
import numpy as np
from PIL import Image
import tempfile
import os

def jpeg_ghost_multiple(image):
    """
    Perform JPEG Ghost analysis on the provided PIL image and return the processed image.

    :param image: PIL.Image.Image object.
    :return: PIL.Image.Image object of the last processed result.
    """
    print("Analyzing JPEG Ghosts...")

    # Convert PIL Image to OpenCV format
    image_cv = np.array(image)
    if len(image_cv.shape) == 3 and image_cv.shape[2] == 3:
        image_cv = cv.cvtColor(image_cv, cv.COLOR_RGB2BGR)

    quality = 60
    smoothing_b = 17
    offset = int((smoothing_b - 1) / 2)
    height, width, channels = image_cv.shape

    # Create a temporary file to save JPEG images
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        save_file_name = temp_file.name

    # Prepare an empty list to store processed images
    processed_images = []

    for pos_q in range(19):
        encode_param = [int(cv.IMWRITE_JPEG_QUALITY), quality]
        cv.imwrite(save_file_name, image_cv, encode_param)
        img_low = cv.imread(save_file_name)
        img_low_rgb = img_low[:, :, ::-1]

        tmp = (image_cv[:, :, ::-1] - img_low_rgb) ** 2
        kernel = np.ones((smoothing_b, smoothing_b), np.float32) / (smoothing_b ** 2)
        tmp = cv.filter2D(tmp, -1, kernel)
        tmp = np.average(tmp, axis=-1)
        tmp = tmp[offset:(int(height - offset)), offset:(int(width - offset))]
        nomalized = tmp.min() / (tmp.max() - tmp.min())
        dst = tmp - nomalized

        # Convert the processed result to PIL Image
        dst_image = Image.fromarray((dst * 255).astype(np.uint8), mode='L')
        processed_images.append(dst_image)

        quality += 2

    # Clean up temporary file
    os.remove(save_file_name)

    # Return the last processed image
    return processed_images[-1] if processed_images else None
