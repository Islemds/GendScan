from PIL import Image
import numpy as np
import cv2 as cv
import pywt

def noise_inconsistencies(img, block_size=8):
    print("Analyzing noise inconsistencies...")

    if block_size is None:
        block_size = 8

    # Convert image to YCrCb
    imgYCC = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    y, _, _ = cv.split(imgYCC)

    # Perform Discrete Wavelet Transform
    coeffs = pywt.dwt2(y, 'db8')
    cA, (cH, cV, cD) = coeffs

    # Resize cD to be divisible by block_size
    cD = cD[0:(len(cD) // block_size) * block_size,
            0:(len(cD[0]) // block_size) * block_size]
    
    # Create blocks of size block_size x block_size
    block = np.zeros(
        (len(cD) // block_size, len(cD[0]) // block_size, block_size ** 2))

    for i in range(0, len(cD), block_size):
        for j in range(0, len(cD[0]), block_size):
            blockElement = cD[i:i + block_size, j:j + block_size]
            temp = np.reshape(blockElement, (1, 1, block_size ** 2))
            block[i // block_size, j // block_size, :] = temp

    # Compute the absolute value of the blocks
    abs_map = np.absolute(block)
    med_map = np.median(abs_map, axis=2)
    
    # Compute noise map
    noise_map = np.divide(med_map, 0.6745)

    # Normalize the noise map to 0-255 for display
    noise_map_normalized = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())
    noise_map_image = (noise_map_normalized * 255).astype(np.uint8)

    # Convert the noise map to a PIL image for display
    noise_map_image = Image.fromarray(noise_map_image)

    print("Noise inconsistencies analysis completed.")
    return noise_map_image
