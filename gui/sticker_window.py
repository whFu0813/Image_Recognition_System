# gui/sticker_window.py

import cv2
import numpy as np
from PyQt5.QtCore import QPoint, QEvent, Qt
from PyQt5.QtWidgets import (QWidget, QLabel, QHBoxLayout, QPushButton,
                             QVBoxLayout, QMessageBox, QCheckBox, QScrollArea, QSlider)

from gui.ui_utils import cv2_to_qpixmap


class StickerWindow(QWidget):
    def __init__(self, image, mask, callback):
        super().__init__()
        self.setWindowTitle("贴纸拖动")
        self.image = image
        self.mask = mask
        self.callback = callback
        self.rotation_angle = 0  # 角度（度数）
        self.scale_factor = 1.0  # 缩放倍数

        # 提取前景贴纸与其在原图中的位置
        self.fg, self.fg_mask, self.orig_pos = self.extract_foreground()
        self.bg = image.copy()

        self.drag_pos = QPoint(*self.orig_pos)  # 初始贴纸位置设为原位
        self.offset = QPoint(0, 0)
        self.dragging = False

        self.cut_mode_checkbox = QCheckBox("剪切模式（用白色填原位）")

        self.rotation_slider = QSlider(Qt.Horizontal)
        self.rotation_slider.setRange(-180, 180)
        self.rotation_slider.setValue(0)
        self.rotation_slider.valueChanged.connect(self.on_transform_changed)

        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setRange(10, 300)
        self.scale_slider.setValue(100)
        self.scale_slider.valueChanged.connect(self.on_transform_changed)

        self.label = QLabel(self)
        self.label.setFixedSize(self.bg.shape[1], self.bg.shape[0])
        self.label.setPixmap(self.get_overlay_pixmap())
        self.label.installEventFilter(self)

        # 滚动区域支持大图
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.label)

        btn_layout = QHBoxLayout()
        apply_btn = QPushButton("应用")
        apply_btn.clicked.connect(self.apply)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.cut_mode_checkbox)
        btn_layout.addWidget(apply_btn)
        btn_layout.addWidget(cancel_btn)

        rotation_layout = QHBoxLayout()
        rotation_layout.addWidget(QLabel("旋转角度"))
        rotation_layout.addWidget(self.rotation_slider)

        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("缩放比例"))
        scale_layout.addWidget(self.scale_slider)

        transform_layout = QVBoxLayout()
        transform_layout.addLayout(rotation_layout)
        transform_layout.addLayout(scale_layout)

        layout = QVBoxLayout()
        layout.addWidget(scroll)
        layout.addLayout(transform_layout)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

        # 初始窗口尺寸
        self.resize(self.bg.shape[1] + 100, self.bg.shape[0] + 200)

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

    def on_transform_changed(self):
        self.rotation_angle = self.rotation_slider.value()
        self.scale_factor = self.scale_slider.value() / 100.0
        self.label.setPixmap(self.get_overlay_pixmap())
        self.label.repaint()

    def transform_fg(self):
        if self.fg is None or self.fg_mask is None:
            return None, None

        scale = self.scale_factor
        fg_scaled = cv2.resize(self.fg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        mask_scaled = cv2.resize(self.fg_mask, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        angle = self.rotation_angle
        center = (fg_scaled.shape[1] // 2, fg_scaled.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int(fg_scaled.shape[0] * sin + fg_scaled.shape[1] * cos)
        new_h = int(fg_scaled.shape[0] * cos + fg_scaled.shape[1] * sin)

        M[0, 2] += (new_w // 2) - center[0]
        M[1, 2] += (new_h // 2) - center[1]

        fg_rotated = cv2.warpAffine(fg_scaled, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        mask_rotated = cv2.warpAffine(mask_scaled, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=0)

        return fg_rotated, mask_rotated

    def get_overlay_pixmap(self):
        display = self.bg.copy()

        if self.fg is None or self.fg_mask is None:
            return cv2_to_qpixmap(display)

        # 获取拖动位置
        x, y = self.drag_pos.x(), self.drag_pos.y()
        h, w = self.fg.shape[:2]

        # 剪切模式下，用白色填原贴纸区域
        if self.cut_mode_checkbox.isChecked():
            x0, y0 = self.orig_pos
            h, w = self.fg.shape[:2]
            white_patch = np.ones_like(self.fg) * 255
            orig_area = display[y0:y0 + h, x0:x0 + w]
            mask_rgb = cv2.cvtColor(self.fg_mask, cv2.COLOR_GRAY2BGR)
            np.putmask(orig_area, mask_rgb > 0, white_patch)
            display[y0:y0 + h, x0:x0 + w] = orig_area

        fg_trans, mask_trans = self.transform_fg()
        if fg_trans is None:
            return cv2_to_qpixmap(display)

        x, y = self.drag_pos.x(), self.drag_pos.y()
        h, w = fg_trans.shape[:2]

        x_start, y_start = max(0, x), max(0, y)
        x_end, y_end = min(display.shape[1], x + w), min(display.shape[0], y + h)

        if x_start >= x_end or y_start >= y_end:
            return cv2_to_qpixmap(display)

        roi = display[y_start:y_end, x_start:x_end]
        x_off, y_off = x_start - x, y_start - y
        roi_w, roi_h = x_end - x_start, y_end - y_start

        patch = fg_trans[y_off:y_off + roi_h, x_off:x_off + roi_w]
        patch_mask = cv2.cvtColor(mask_trans[y_off:y_off + roi_h, x_off:x_off + roi_w], cv2.COLOR_GRAY2BGR)

        np.putmask(roi, patch_mask > 0, patch)
        display[y_start:y_end, x_start:x_end] = roi

        return cv2_to_qpixmap(display)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress:
            self.dragging = True
            self.offset = event.pos() - self.drag_pos
            self.setCursor(Qt.ClosedHandCursor)
            return True
        elif event.type() == QEvent.MouseMove and self.dragging:
            self.drag_pos = event.pos() - self.offset
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

        # 剪切原贴纸位置
        if self.cut_mode_checkbox.isChecked():
            x0, y0 = self.orig_pos
            h, w = self.fg.shape[:2]
            white_patch = np.ones_like(self.fg) * 255
            orig_area = result[y0:y0 + h, x0:x0 + w]
            mask_rgb = cv2.cvtColor(self.fg_mask, cv2.COLOR_GRAY2BGR)
            np.putmask(orig_area, mask_rgb > 0, white_patch)
            result[y0:y0 + h, x0:x0 + w] = orig_area

        # 裁剪后贴图
        fg_trans, mask_trans = self.transform_fg()
        if fg_trans is not None:
            x, y = self.drag_pos.x(), self.drag_pos.y()
            h, w = fg_trans.shape[:2]

            x_start, y_start = max(0, x), max(0, y)
            x_end, y_end = min(result.shape[1], x + w), min(result.shape[0], y + h)

            if x_start < x_end and y_start < y_end:
                roi = result[y_start:y_end, x_start:x_end]
                x_off, y_off = x_start - x, y_start - y
                roi_w, roi_h = x_end - x_start, y_end - y_start

                patch = fg_trans[y_off:y_off + roi_h, x_off:x_off + roi_w]
                patch_mask = cv2.cvtColor(mask_trans[y_off:y_off + roi_h, x_off:x_off + roi_w], cv2.COLOR_GRAY2BGR)
                np.putmask(roi, patch_mask > 0, patch)
                result[y_start:y_end, x_start:x_end] = roi

        self.callback(result)
        self.close()


