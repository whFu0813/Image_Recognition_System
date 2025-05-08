from PyQt5.QtGui import QImage, QPixmap
import cv2

def cv2_to_qpixmap(cv_img):
    if len(cv_img.shape) == 2:
        q_img = QImage(cv_img.data, cv_img.shape[1], cv_img.shape[0], cv_img.strides[0], QImage.Format_Grayscale8)
    else:
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        q_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(q_img)
