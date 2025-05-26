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
    """将二值图转换为彩色显示的Mask图"""
    mask_img = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)
    mask_img[binary_mask == 255] = (255, 255, 255)
    return mask_img

def edit_image(original, mask):
    edited = original.copy()
    edited[mask == 0] = [255, 255, 255]
    return edited

def measure_features(binary_image):
    # 特征测量
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = 0, 0
        results.append({"area": area, "perimeter": perimeter, "centroid": (cx, cy)})
    return results
