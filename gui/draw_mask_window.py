# gui/draw_mask_window.py

from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QSlider, QSpinBox
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
        self.mode = 'draw'  # 模式：'draw' 或 'erase'
        self.history = []  # 保存mask历史，用于撤销

        self.brush_size = 10  # 默认笔刷大小

        # 画布（显示原图 + 绘制）
        self.canvas_label = QLabel()
        self.update_canvas()

        # 按钮与滑动条
        self.apply_btn = QPushButton("应用")
        self.apply_btn.clicked.connect(self.apply_changes)

        self.erase_btn = QPushButton("橡皮擦模式")
        self.erase_btn.setCheckable(True)
        self.erase_btn.clicked.connect(self.toggle_erase_mode)

        self.undo_btn = QPushButton("撤销")
        self.undo_btn.clicked.connect(self.undo_last)

        # 笔刷大小滑动条与数值框
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(1, 50)
        self.brush_slider.setValue(self.brush_size)
        self.brush_slider.valueChanged.connect(self.change_brush_size)

        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(1, 50)
        self.brush_spin.setValue(self.brush_size)
        self.brush_spin.valueChanged.connect(self.change_brush_size)

        self.brush_slider.valueChanged.connect(self.brush_spin.setValue)
        self.brush_spin.valueChanged.connect(self.brush_slider.setValue)

        # 布局
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.erase_btn)
        btn_layout.addWidget(self.undo_btn)
        btn_layout.addWidget(self.apply_btn)

        brush_layout = QHBoxLayout()
        brush_layout.addWidget(QLabel("笔刷大小:"))
        brush_layout.addWidget(self.brush_slider)
        brush_layout.addWidget(self.brush_spin)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas_label)
        layout.addLayout(brush_layout)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

        # 鼠标事件
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
            self.history.append(self.mask.copy())  # 保存历史

    def mouse_move_event(self, event):
        if self.drawing:
            # 缩放笔刷大小以适应 QLabel 显示尺寸
            label_w = self.canvas_label.width()
            label_h = self.canvas_label.height()
            img_h, img_w = self.mask.shape
            scale_x = label_w / img_w
            scale_y = label_h / img_h
            avg_scale = (scale_x + scale_y) / 2

            display_brush_size = max(1, int(self.brush_size * avg_scale))  # 缩放后的显示尺寸

            # 绘图视觉反馈
            painter = QPainter(self.canvas_label.pixmap())
            if self.mode == 'draw':
                pen = QPen(Qt.red, display_brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                color = 255
            else:  # 橡皮擦
                pen = QPen(Qt.black, display_brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                color = 0

            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            painter.end()

            self.update_mask_line(self.last_point, event.pos(), color)
            self.last_point = event.pos()
            self.canvas_label.update()

    def mouse_release_event(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def update_mask_line(self, start, end, color):
        # 根据起止点，在mask图中画线（转换为图像坐标）
        h, w = self.mask.shape
        label_w = self.canvas_label.width()
        label_h = self.canvas_label.height()

        x1, y1 = int(start.x() * w / self.canvas_label.width()), int(start.y() * h / self.canvas_label.height())
        x2, y2 = int(end.x() * w / self.canvas_label.width()), int(end.y() * h / self.canvas_label.height())

        # 缩放笔刷大小到 mask 尺寸
        scale_x = w / label_w
        scale_y = h / label_h
        avg_scale = (scale_x + scale_y) / 2
        thickness = max(1, int(self.brush_size * avg_scale))  # 防止为 0

        cv2.line(self.mask, (x1, y1), (x2, y2), color=color, thickness=thickness)

    def toggle_erase_mode(self):
        if self.erase_btn.isChecked():
            self.mode = 'erase'
            self.erase_btn.setText("绘制模式")
        else:
            self.mode = 'draw'
            self.erase_btn.setText("橡皮擦模式")

    def undo_last(self):
        if self.history:
            self.mask = self.history.pop()
            self.update_canvas()

    def change_brush_size(self, value):
        self.brush_size = value

    def apply_changes(self):
        self.callback(self.mask)  # 调用主窗口提供的函数
        self.close()
