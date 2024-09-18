from PIL import Image
import numpy as np
import cv2 as cv
import os

def ela(image, quality=90, block_size=8):
    print("Analyzing...")

    img_rgb = image[:, :, ::-1]  # Convert BGR to RGB

    # Create a temporary filename for saving the image
    save_file_name = "temp_image.jpg"

    multiplier = 15
    flatten = True

    # Resave the image with the new quality
    encode_param = [int(cv.IMWRITE_JPEG_QUALITY), quality]
    cv.imwrite(save_file_name, image, encode_param)

    # Load resaved image
    img_low = cv.imread(save_file_name)
    img_low = img_low[:, :, ::-1]  # Convert BGR to RGB

    ela_map = np.zeros((img_rgb.shape[0], img_rgb.shape[1], 3))
    ela_map = np.absolute(1.0 * img_rgb - 1.0 * img_low) * multiplier

    if flatten:
        ela_map = np.average(ela_map, axis=-1)
        ela_map = np.clip(ela_map, 0, 255).astype(np.uint8)

    # Clean up the temporary file
    os.remove(save_file_name)

    print("Done")

    # Convert the ELA result to a PIL Image
    ela_image_pil = Image.fromarray(ela_map)
    
    # Set format explicitly
    ela_image_pil.format = "JPEG"

    return ela_image_pil
