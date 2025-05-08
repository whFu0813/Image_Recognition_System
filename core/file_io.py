import cv2
from PIL import Image
import numpy as np

def read_image(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return img

def save_image(path, image):
    ext = path.split('.')[-1]
    cv2.imencode(f'.{ext}', image)[1].tofile(path)

def capture_from_camera(index=0):
    cap = cv2.VideoCapture(index)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None
