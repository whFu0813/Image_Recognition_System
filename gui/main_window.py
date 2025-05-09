import sys
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QComboBox, QSlider,
    QFileDialog, QHBoxLayout, QMessageBox, QScrollArea, QSplitter, QSizePolicy
)
from PyQt5.QtCore import Qt
from core import file_io, segmentation, recognition_edit
from gui.ui_utils import cv2_to_qpixmap
import cv2

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像识别系统")
        self.resize(1200, 800)

        self.image = None
        self.mask = None
        self.result = None

        self.image_label = QLabel("原图")
        self.mask_label = QLabel("Mask图")
        self.result_label = QLabel("处理结果")

        for label in (self.image_label, self.mask_label, self.result_label):
            label.setAlignment(Qt.AlignCenter)
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 使用滚动区域包裹图片标签
        scroll_image = QScrollArea()
        scroll_image.setWidgetResizable(True)
        scroll_image.setWidget(self.image_label)

        scroll_mask = QScrollArea()
        scroll_mask.setWidgetResizable(True)
        scroll_mask.setWidget(self.mask_label)

        scroll_result = QScrollArea()
        scroll_result.setWidgetResizable(True)
        scroll_result.setWidget(self.result_label)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll_image)
        splitter.addWidget(scroll_mask)
        splitter.addWidget(scroll_result)

        self.btn_open = QPushButton("打开图像")
        self.btn_segment = QPushButton("阈值分割+抠图")
        self.btn_edit = QPushButton("编辑目标")
        self.btn_save = QPushButton("保存结果")
        self.btn_morph = QPushButton("应用形态学处理")

        self.btn_open.clicked.connect(self.open_image)
        self.btn_segment.clicked.connect(self.segment_and_mask)
        self.btn_edit.clicked.connect(self.edit_target)
        self.btn_save.clicked.connect(self.save_result)
        self.btn_morph.clicked.connect(self.apply_morphology)

        self.morph_label = QLabel("形态学处理：")
        self.morph_combo = QComboBox()
        self.morph_combo.addItems(["腐蚀", "膨胀", "开运算", "闭运算"])

        self.kernel_slider = QSlider(Qt.Horizontal)
        self.kernel_slider.setMinimum(1)
        self.kernel_slider.setMaximum(21)
        self.kernel_slider.setValue(3)
        self.kernel_slider.setTickInterval(2)
        self.kernel_slider.setTickPosition(QSlider.TicksBelow)

        btn_layout = QHBoxLayout()
        for btn in [self.btn_open, self.btn_segment, self.btn_edit, self.btn_save]:
            btn_layout.addWidget(btn)

        morph_layout = QHBoxLayout()
        morph_layout.addWidget(self.morph_label)
        morph_layout.addWidget(self.morph_combo)
        morph_layout.addWidget(QLabel("核大小"))
        morph_layout.addWidget(self.kernel_slider)
        morph_layout.addWidget(self.btn_morph)

        layout = QVBoxLayout()
        layout.addLayout(btn_layout)
        layout.addWidget(splitter)
        layout.addLayout(morph_layout)

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

    def apply_morphology(self):
        if self.image is None:
            QMessageBox.warning(self, "提示", "请先加载图像")
            return

        method = self.morph_combo.currentText()
        ksize = self.kernel_slider.value()
        if ksize % 2 == 0:
            ksize += 1

        binary = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))

        if method == "腐蚀":
            result = cv2.erode(binary, kernel)
        elif method == "膨胀":
            result = cv2.dilate(binary, kernel)
        elif method == "开运算":
            result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        elif method == "闭运算":
            result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        else:
            result = binary

        morph_result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        self.result = morph_result
        self.result_label.setPixmap(cv2_to_qpixmap(morph_result))