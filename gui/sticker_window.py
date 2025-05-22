# gui/sticker_window.py

import cv2
import numpy as np
from PyQt5.QtCore import QPoint, QEvent, Qt
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout, QPushButton, QVBoxLayout, QMessageBox

from gui.ui_utils import cv2_to_qpixmap


class StickerWindow(QWidget):
    def __init__(self, image, mask, callback):
        super().__init__()
        self.setWindowTitle("贴纸拖动")
        self.image = image
        self.mask = mask
        self.callback = callback

        self.fg = self.extract_foreground()
        self.bg = image.copy()

        self.drag_pos = QPoint(0, 0)
        self.offset = QPoint(0, 0)
        self.dragging = False

        self.label = QLabel(self)
        self.label.setPixmap(self.get_overlay_pixmap())
        self.label.setFixedSize(self.bg.shape[1], self.bg.shape[0])
        self.label.installEventFilter(self)

        btn_layout = QHBoxLayout()
        apply_btn = QPushButton("应用")
        apply_btn.clicked.connect(self.apply)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.close)
        btn_layout.addWidget(apply_btn)
        btn_layout.addWidget(cancel_btn)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

        self.setFixedSize(self.bg.shape[1], self.bg.shape[0] + 50)  # 加按钮区域高度

    def extract_foreground(self):
        fg = cv2.bitwise_and(self.image, self.image, mask=self.mask)
        return fg

    def get_overlay_pixmap(self):
        display = self.bg.copy()
        x, y = self.drag_pos.x(), self.drag_pos.y()
        h, w = self.fg.shape[:2]

        # 限制贴图不越界
        x = max(0, min(x, display.shape[1] - w))
        y = max(0, min(y, display.shape[0] - h))
        self.drag_pos = QPoint(x, y)

        fg_area = display[y:y + h, x:x + w]
        mask_rgb = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
        fg_alpha = cv2.bitwise_and(self.fg, mask_rgb)
        np.putmask(fg_area, mask_rgb > 0, fg_alpha)
        display[y:y + h, x:x + w] = fg_area

        # 可选：叠加一个红框用于提示贴图区域
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 2)

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
            return True
        elif event.type() == QEvent.MouseButtonRelease:
            self.dragging = False
            self.setCursor(Qt.ArrowCursor)
            return True
        return False

    def apply(self):
        result = self.bg.copy()
        x, y = self.drag_pos.x(), self.drag_pos.y()
        h, w = self.fg.shape[:2]

        if x < 0 or y < 0 or x + w > result.shape[1] or y + h > result.shape[0]:
            QMessageBox.warning(self, "超出范围", "贴纸部分越界，无法应用")
            return

        fg_area = result[y:y + h, x:x + w]
        mask_rgb = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
        fg_alpha = cv2.bitwise_and(self.fg, mask_rgb)
        np.putmask(fg_area, mask_rgb > 0, fg_alpha)
        result[y:y + h, x:x + w] = fg_area

        self.callback(result)
        self.close()
