from PyQt5.QtGui import QImage, QPixmap
import cv2

def cv2_to_qpixmap(cv_img):
    height, width = cv_img.shape[:2]
    if cv_img.shape[2] == 4:  # RGBA 图像
        qimg = QImage(cv_img.data, width, height, width * 4, QImage.Format_RGBA8888)
    else:
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_image.data, width, height, width * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)
