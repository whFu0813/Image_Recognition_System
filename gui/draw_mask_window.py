# gui/draw_mask_window.py

from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage
from PyQt5.QtCore import Qt, QPoint
import numpy as np
import cv2

class DrawMaskWindow(QWidget):
    def __init__(self, image, initial_mask, callback):
        super().__init__()
        self.setWindowTitle("绘制蒙版")
        self.image = image.copy()
        self.mask = initial_mask.copy()
        self.callback = callback  # 回调函数，主窗口提供

        self.drawing = False
        self.last_point = QPoint()

        # 画布（显示原图 + 绘制）
        self.canvas_label = QLabel()
        self.update_canvas()

        # 应用按钮
        self.apply_btn = QPushButton("应用")
        self.apply_btn.clicked.connect(self.apply_changes)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas_label)
        layout.addWidget(self.apply_btn)
        self.setLayout(layout)

        self.canvas_label.mousePressEvent = self.mouse_press_event
        self.canvas_label.mouseMoveEvent = self.mouse_move_event
        self.canvas_label.mouseReleaseEvent = self.mouse_release_event

    def update_canvas(self):
        # 可视化：原图 + mask为红色透明覆盖
        overlay = self.image.copy()
        red_mask = np.zeros_like(overlay)
        red_mask[:, :, 2] = 255  # R通道全红
        alpha_mask = self.mask.astype(bool)
        overlay[alpha_mask] = cv2.addWeighted(overlay, 0.5, red_mask, 0.5, 0)[alpha_mask]
        qpix = QPixmap.fromImage(QImage(overlay.data, overlay.shape[1], overlay.shape[0], overlay.strides[0], QImage.Format_BGR888))
        self.canvas_label.setPixmap(qpix)

    def mouse_press_event(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouse_move_event(self, event):
        if self.drawing:
            painter = QPainter(self.canvas_label.pixmap())
            pen = QPen(Qt.red, 10, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            painter.end()
            self.update_mask_line(self.last_point, event.pos())
            self.last_point = event.pos()
            self.canvas_label.update()

    def mouse_release_event(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def update_mask_line(self, start, end):
        # 根据起止点，在mask图中画线（转换为图像坐标）
        h, w = self.mask.shape
        x1, y1 = int(start.x() * w / self.canvas_label.width()), int(start.y() * h / self.canvas_label.height())
        x2, y2 = int(end.x() * w / self.canvas_label.width()), int(end.y() * h / self.canvas_label.height())
        cv2.line(self.mask, (x1, y1), (x2, y2), color=255, thickness=10)

    def apply_changes(self):
        self.callback(self.mask)  # 调用主窗口提供的函数
        self.close()
