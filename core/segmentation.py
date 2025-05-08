import cv2
import numpy as np

def region_growing(image, seed_point, threshold=10):
    visited = np.zeros_like(image, np.uint8)
    h, w = image.shape[:2]
    stack = [seed_point]
    region_val = int(image[seed_point[1], seed_point[0]])
    while stack:
        x, y = stack.pop()
        if visited[y, x] == 0 and abs(int(image[y, x]) - region_val) < threshold:
            visited[y, x] = 255
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    stack.append((nx, ny))
    return visited

def edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    grad = cv2.magnitude(grad_x, grad_y)
    return np.uint8(np.clip(grad, 0, 255))

def threshold_segmentation(image, thresh_val=127):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    return binary
