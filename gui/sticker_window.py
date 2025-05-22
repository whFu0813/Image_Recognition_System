# gui/sticker_window.py

import cv2
import numpy as np
from PyQt5.QtCore import QPoint, QEvent, Qt
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout, QPushButton, QVBoxLayout, QMessageBox, QCheckBox

from gui.ui_utils import cv2_to_qpixmap


class StickerWindow(QWidget):
    def __init__(self, image, mask, callback):
        super().__init__()
        self.setWindowTitle("贴纸拖动")
        self.image = image
        self.mask = mask
        self.callback = callback

        # 提取前景贴纸与其在原图中的位置
        self.fg, self.fg_mask, self.orig_pos = self.extract_foreground()
        self.bg = image.copy()

        self.drag_pos = QPoint(*self.orig_pos)  # 初始贴纸位置设为原位
        self.offset = QPoint(0, 0)
        self.dragging = False

        self.label = QLabel(self)

        self.cut_mode_checkbox = QCheckBox("剪切模式（用白色填原位）")

        btn_layout = QHBoxLayout()
        apply_btn = QPushButton("应用")
        apply_btn.clicked.connect(self.apply)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.cut_mode_checkbox)
        btn_layout.addWidget(apply_btn)
        btn_layout.addWidget(cancel_btn)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

        # 设置窗口大小并渲染初始图像
        self.label.setFixedSize(self.bg.shape[1], self.bg.shape[0])
        self.label.setPixmap(self.get_overlay_pixmap())
        self.setFixedSize(self.bg.shape[1], self.bg.shape[0] + 50)
        self.label.installEventFilter(self)

        #鼠标跟踪
        self.label.setMouseTracking(True)
        self.setMouseTracking(True)

    def extract_foreground(self):
        # 获取 mask 的非零边界矩形
        y_indices, x_indices = np.where(self.mask > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None, None, None  # 空 mask

        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()

        self.mask_roi = self.mask[y_min:y_max + 1, x_min:x_max + 1]
        fg = self.image[y_min:y_max + 1, x_min:x_max + 1]
        return fg, self.mask_roi, (x_min, y_min)

    def get_overlay_pixmap(self):
        display = self.bg.copy()

        if self.fg is None or self.fg_mask is None:
            return cv2_to_qpixmap(display)

        # 剪切模式下，用白色填原贴纸区域
        if self.cut_mode_checkbox.isChecked():
            x0, y0 = self.orig_pos
            h, w = self.fg.shape[:2]
            white_patch = np.ones_like(self.fg) * 255
            orig_area = display[y0:y0 + h, x0:x0 + w]
            mask_rgb = cv2.cvtColor(self.fg_mask, cv2.COLOR_GRAY2BGR)
            np.putmask(orig_area, mask_rgb > 0, white_patch)
            display[y0:y0 + h, x0:x0 + w] = orig_area

        # 在拖动位置合成贴纸
        x, y = self.drag_pos.x(), self.drag_pos.y()
        h, w = self.fg.shape[:2]

        if 0 <= x < display.shape[1] - w and 0 <= y < display.shape[0] - h:
            roi = display[y:y + h, x:x + w]
            mask_rgb = cv2.cvtColor(self.fg_mask, cv2.COLOR_GRAY2BGR)
            fg_alpha = cv2.bitwise_and(self.fg, mask_rgb)
            np.putmask(roi, mask_rgb > 0, fg_alpha)
            display[y:y + h, x:x + w] = roi

        return cv2_to_qpixmap(display)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress:
            self.dragging = True
            self.offset = event.pos() - self.drag_pos
            self.setCursor(Qt.ClosedHandCursor)
            return True
        elif event.type() == QEvent.MouseMove and self.dragging:
            self.drag_pos = event.pos() - self.offset

            # 限制贴纸不越界（左上角为起点）
            self.drag_pos.setX(max(0, min(self.drag_pos.x(), self.bg.shape[1] - self.fg.shape[1])))
            self.drag_pos.setY(max(0, min(self.drag_pos.y(), self.bg.shape[0] - self.fg.shape[0])))

            self.label.setPixmap(self.get_overlay_pixmap())
            self.label.repaint()  # 强制刷新
            print("拖动坐标：", self.drag_pos)

            return True
        elif event.type() == QEvent.MouseButtonRelease:
            self.dragging = False
            self.setCursor(Qt.ArrowCursor)
            return True
        return False

    def apply(self):
        result = self.bg.copy()

        if self.fg is None or self.fg_mask is None:
            self.callback(result)
            self.close()
            return

        x, y = self.drag_pos.x(), self.drag_pos.y()
        h, w = self.fg.shape[:2]

        # 裁剪前景区域避免越界
        if 0 <= x < result.shape[1] - w and 0 <= y < result.shape[0] - h:
            roi = result[y:y + h, x:x + w]
            mask_rgb = cv2.cvtColor(self.fg_mask, cv2.COLOR_GRAY2BGR)
            fg_alpha = cv2.bitwise_and(self.fg, mask_rgb)
            np.putmask(roi, mask_rgb > 0, fg_alpha)
            result[y:y + h, x:x + w] = roi

            # 剪切原贴纸位置
            if self.cut_mode_checkbox.isChecked():
                x0, y0 = self.orig_pos
                h, w = self.fg.shape[:2]
                white_patch = np.ones_like(self.fg) * 255
                orig_area = result[y0:y0 + h, x0:x0 + w]
                mask_rgb = cv2.cvtColor(self.fg_mask, cv2.COLOR_GRAY2BGR)
                np.putmask(orig_area, mask_rgb > 0, white_patch)
                result[y0:y0 + h, x0:x0 + w] = orig_area

        self.callback(result)
        self.close()


