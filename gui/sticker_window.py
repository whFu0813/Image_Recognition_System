import cv2
import numpy as np
from PyQt5.QtCore import QPoint, QEvent, Qt
from PyQt5.QtWidgets import (
    QWidget, QLabel, QHBoxLayout, QPushButton, QVBoxLayout, QMessageBox,
    QCheckBox, QScrollArea, QSlider
)
from gui.ui_utils import cv2_to_qpixmap


class Sticker:
    def __init__(self, fg, mask, pos: QPoint):
        self.fg = fg
        self.mask = mask
        self.pos = QPoint(pos)
        self.rotation = 0
        self.scale = 1.0

    def transformed(self):
        # 缩放
        fg_scaled = cv2.resize(self.fg, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        mask_scaled = cv2.resize(self.mask, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)

        # 旋转
        angle = self.rotation
        center = (fg_scaled.shape[1] // 2, fg_scaled.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int(fg_scaled.shape[0] * sin + fg_scaled.shape[1] * cos)
        new_h = int(fg_scaled.shape[0] * cos + fg_scaled.shape[1] * sin)
        M[0, 2] += (new_w // 2) - center[0]
        M[1, 2] += (new_h // 2) - center[1]
        fg_rot = cv2.warpAffine(fg_scaled, M, (new_w, new_h), flags=cv2.INTER_LINEAR)
        mask_rot = cv2.warpAffine(mask_scaled, M, (new_w, new_h), flags=cv2.INTER_NEAREST)
        return fg_rot, mask_rot


class StickerWindow(QWidget):
    def __init__(self, image, mask, callback):
        super().__init__()
        self.setWindowTitle("贴纸拖动")
        self.image = image
        self.mask = mask
        self.callback = callback
        self.bg = image.copy()

        self.fg, self.fg_mask, self.orig_pos = self.extract_foreground()
        self.stickers = []
        self.selected_index = -1
        self.dragging = False
        self.offset = QPoint()

        self.cut_mode_checkbox = QCheckBox("剪切模式")

        self.rotation_slider = QSlider(Qt.Horizontal)
        self.rotation_slider.setRange(-180, 180)
        self.rotation_slider.valueChanged.connect(self.on_transform_changed)

        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setRange(10, 300)
        self.scale_slider.valueChanged.connect(self.on_transform_changed)

        self.label = QLabel(self)
        self.label.setPixmap(self.get_overlay_pixmap())
        self.label.installEventFilter(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.label)

        layout = QVBoxLayout()
        transform_layout = QHBoxLayout()
        transform_layout.addWidget(QLabel("旋转角度"))
        transform_layout.addWidget(self.rotation_slider)
        transform_layout.addWidget(QLabel("缩放比例"))
        transform_layout.addWidget(self.scale_slider)
        layout.addLayout(transform_layout)

        btn_layout = QHBoxLayout()
        apply_btn = QPushButton("应用")
        apply_btn.clicked.connect(self.apply)
        delete_btn = QPushButton("删除当前贴纸")
        delete_btn.clicked.connect(self.delete_selected)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.close)

        btn_layout.addWidget(self.cut_mode_checkbox)
        btn_layout.addWidget(apply_btn)
        btn_layout.addWidget(delete_btn)
        btn_layout.addWidget(cancel_btn)

        layout.addWidget(scroll)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

        self.resize(self.bg.shape[1] + 100, self.bg.shape[0] + 200)
        self.label.setMouseTracking(True)

    def extract_foreground(self):
        y_idx, x_idx = np.where(self.mask > 0)
        if len(x_idx) == 0 or len(y_idx) == 0:
            return None, None, None
        x_min, x_max = x_idx.min(), x_idx.max()
        y_min, y_max = y_idx.min(), y_idx.max()
        mask_roi = self.mask[y_min:y_max + 1, x_min:x_max + 1]
        fg = self.image[y_min:y_max + 1, x_min:x_max + 1]
        return fg, mask_roi, QPoint(x_min, y_min)

    def add_sticker(self, pos: QPoint):
        new = Sticker(self.fg, self.fg_mask, pos)
        self.stickers.append(new)
        self.selected_index = len(self.stickers) - 1
        self.update_sliders()
        self.label.setPixmap(self.get_overlay_pixmap())

    def delete_selected(self):
        if self.selected_index >= 0:
            del self.stickers[self.selected_index]
            self.selected_index = len(self.stickers) - 1
            self.update_sliders()
            self.label.setPixmap(self.get_overlay_pixmap())

    def on_transform_changed(self):
        if self.selected_index >= 0:
            self.stickers[self.selected_index].rotation = self.rotation_slider.value()
            self.stickers[self.selected_index].scale = self.scale_slider.value() / 100.0
            self.label.setPixmap(self.get_overlay_pixmap())

    def update_sliders(self):
        if self.selected_index >= 0:
            s = self.stickers[self.selected_index]
            self.rotation_slider.blockSignals(True)
            self.scale_slider.blockSignals(True)
            self.rotation_slider.setValue(s.rotation)
            self.scale_slider.setValue(int(s.scale * 100))
            self.rotation_slider.blockSignals(False)
            self.scale_slider.blockSignals(False)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress:
            clicked = event.pos()
            for i in reversed(range(len(self.stickers))):
                s = self.stickers[i]
                fg_t, _ = s.transformed()
                x, y = s.pos.x(), s.pos.y()
                w, h = fg_t.shape[1], fg_t.shape[0]
                if x <= clicked.x() <= x + w and y <= clicked.y() <= y + h:
                    self.selected_index = i
                    self.offset = clicked - s.pos
                    self.dragging = True
                    self.update_sliders()
                    return True
            # 点击空白区域，添加新贴纸
            self.add_sticker(clicked)
            self.dragging = True
            self.offset = QPoint(0, 0)
            return True
        elif event.type() == QEvent.MouseMove and self.dragging:
            if self.selected_index >= 0:
                self.stickers[self.selected_index].pos = event.pos() - self.offset
                self.label.setPixmap(self.get_overlay_pixmap())
            return True
        elif event.type() == QEvent.MouseButtonRelease:
            self.dragging = False
            return True
        return False

    def get_overlay_pixmap(self):
        display = self.bg.copy()

        if self.cut_mode_checkbox.isChecked():
            x0, y0 = self.orig_pos.x(), self.orig_pos.y()
            h, w = self.fg.shape[:2]
            patch = np.ones_like(self.fg) * 255
            mask_rgb = cv2.cvtColor(self.fg_mask, cv2.COLOR_GRAY2BGR)
            np.putmask(display[y0:y0 + h, x0:x0 + w], mask_rgb > 0, patch)

        for s in self.stickers:
            fg_t, mask_t = s.transformed()
            x, y = s.pos.x(), s.pos.y()
            h, w = fg_t.shape[:2]
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + w, display.shape[1]), min(y + h, display.shape[0])
            if x1 >= x2 or y1 >= y2:
                continue
            roi = display[y1:y2, x1:x2]
            x_off, y_off = x1 - x, y1 - y
            patch = fg_t[y_off:y_off + (y2 - y1), x_off:x_off + (x2 - x1)]
            mask = cv2.cvtColor(mask_t[y_off:y_off + (y2 - y1), x_off:x_off + (x2 - x1)], cv2.COLOR_GRAY2BGR)
            np.putmask(roi, mask > 0, patch)
            display[y1:y2, x1:x2] = roi

        return cv2_to_qpixmap(display)

    def apply(self):
        result = self.bg.copy()
        if self.cut_mode_checkbox.isChecked():
            x0, y0 = self.orig_pos.x(), self.orig_pos.y()
            h, w = self.fg.shape[:2]
            patch = np.ones_like(self.fg) * 255
            mask_rgb = cv2.cvtColor(self.fg_mask, cv2.COLOR_GRAY2BGR)
            np.putmask(result[y0:y0 + h, x0:x0 + w], mask_rgb > 0, patch)

        for s in self.stickers:
            fg_t, mask_t = s.transformed()
            x, y = s.pos.x(), s.pos.y()
            h, w = fg_t.shape[:2]
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + w, result.shape[1]), min(y + h, result.shape[0])
            if x1 >= x2 or y1 >= y2:
                continue
            roi = result[y1:y2, x1:x2]
            x_off, y_off = x1 - x, y1 - y
            patch = fg_t[y_off:y_off + (y2 - y1), x_off:x_off + (x2 - x1)]
            mask = cv2.cvtColor(mask_t[y_off:y_off + (y2 - y1), x_off:x_off + (x2 - x1)], cv2.COLOR_GRAY2BGR)
            np.putmask(roi, mask > 0, patch)
            result[y1:y2, x1:x2] = roi

        self.callback(result)
        self.close()
