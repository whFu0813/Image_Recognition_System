import cv2
import numpy as np
from .preprocessing import erode, dilate

def refine_segmentation(binary_image):
    result = dilate(binary_image, 5)
    result = erode(result, 5)
    return result

def generate_mask(image, binary_mask):
    return cv2.bitwise_and(image, image, mask=binary_mask)

def create_mask_image(binary_mask):
    mask_img = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)
    mask_img[binary_mask == 255] = (255, 255, 255)
    return mask_img

def edit_image(original, mask):
    edited = original.copy()
    edited[mask == 0] = [255, 255, 255]
    return edited
