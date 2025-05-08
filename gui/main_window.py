import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QFileDialog, QHBoxLayout, QMessageBox
)
from core import file_io, segmentation, recognition_edit
from gui.ui_utils import cv2_to_qpixmap
import cv2

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像识别系统")
        self.setFixedSize(1000, 600)

        self.image = None
        self.mask = None
        self.result = None

        self.image_label = QLabel("原图")
        self.mask_label = QLabel("Mask图")
        self.result_label = QLabel("处理结果")

        self.btn_open = QPushButton("打开图像")
        self.btn_segment = QPushButton("阈值分割+抠图")
        self.btn_edit = QPushButton("编辑目标")
        self.btn_save = QPushButton("保存结果")

        self.btn_open.clicked.connect(self.open_image)
        self.btn_segment.clicked.connect(self.segment_and_mask)
        self.btn_edit.clicked.connect(self.edit_target)
        self.btn_save.clicked.connect(self.save_result)

        layout = QVBoxLayout()
        btn_layout = QHBoxLayout()
        img_layout = QHBoxLayout()

        btn_layout.addWidget(self.btn_open)
        btn_layout.addWidget(self.btn_segment)
        btn_layout.addWidget(self.btn_edit)
        btn_layout.addWidget(self.btn_save)

        img_layout.addWidget(self.image_label)
        img_layout.addWidget(self.mask_label)
        img_layout.addWidget(self.result_label)

        layout.addLayout(btn_layout)
        layout.addLayout(img_layout)

        self.setLayout(layout)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            self.image = file_io.read_image(file_path)
            self.image_label.setPixmap(cv2_to_qpixmap(self.image))

    def segment_and_mask(self):
        if self.image is None:
            QMessageBox.warning(self, "提示", "请先加载图像")
            return
        binary = segmentation.threshold_segmentation(self.image, 127)
        binary = recognition_edit.refine_segmentation(binary)
        self.mask = recognition_edit.create_mask_image(binary)
        self.mask_label.setPixmap(cv2_to_qpixmap(self.mask))

    def edit_target(self):
        if self.image is None or self.mask is None:
            QMessageBox.warning(self, "提示", "请先生成Mask")
            return
        binary = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
        self.result = recognition_edit.edit_image(self.image, binary)
        self.result_label.setPixmap(cv2_to_qpixmap(self.result))

    def save_result(self):
        if self.result is None:
            QMessageBox.warning(self, "提示", "没有结果可以保存")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "保存图像", "", "PNG Files (*.png);;JPG Files (*.jpg)")
        if file_path:
            file_io.save_image(file_path, self.result)
            QMessageBox.information(self, "保存成功", "图像已保存")
