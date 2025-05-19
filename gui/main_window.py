import sys

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QComboBox, QSlider,
    QFileDialog, QHBoxLayout, QMessageBox, QScrollArea, QSplitter, QSizePolicy
)
from PyQt5.QtCore import Qt
from core import file_io, segmentation, recognition_edit
from core.file_io import read_image
from gui.ui_utils import cv2_to_qpixmap
import cv2

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像识别系统")
        self.resize(1600, 1000)

        self.image = None
        self.mask = None
        self.result = None
        self.latest_binary = None  # 用于编辑阶段的mask/分割结果
        self.drawing_mode = False  # 绘制蒙版模式
        self.drawing_mask = None
        self.last_x, self.last_y = None, None  # 绘制中存储上次鼠标坐标

        self.image_label = QLabel("原图")
        self.mask_label = QLabel("Mask图")
        self.result_label = QLabel("处理结果")

        # 鼠标按下、移动事件
        self.result_label.mousePressEvent = self.mouse_press_event
        self.result_label.mouseMoveEvent = self.mouse_move_event

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
        self.btn_open.clicked.connect(self.open_image)
        self.btn_edit = QPushButton("应用目标抠图")
        self.btn_edit.clicked.connect(self.edit_target)
        self.btn_save = QPushButton("保存结果")
        self.btn_save.clicked.connect(self.save_result)
        self.btn_morph = QPushButton("应用形态学处理")
        self.btn_morph.clicked.connect(self.apply_morphology)
        self.btn_mask = QPushButton("生成 Mask")
        self.btn_mask.clicked.connect(self.generate_mask)
        self.btn_edit_mask = QPushButton("Mask处理")
        self.btn_edit_mask.clicked.connect(self.edit_mask)
        self.btn_edit_bg = QPushButton("背景处理")
        self.btn_edit_bg.clicked.connect(self.edit_background)
        self.btn_confirm_draw = QPushButton("确认绘制")
        self.btn_confirm_draw.clicked.connect(self.confirm_drawing)

        self.morph_label = QLabel("形态学处理：")
        self.morph_combo = QComboBox()
        self.morph_combo.addItems(["腐蚀", "膨胀", "开运算", "闭运算"])

        self.kernel_slider = QSlider(Qt.Horizontal)
        self.kernel_slider.setMinimum(1)
        self.kernel_slider.setMaximum(21)
        self.kernel_slider.setValue(3)
        self.kernel_slider.setTickInterval(2)
        self.kernel_slider.setTickPosition(QSlider.TicksBelow)

        self.segment_combo = QComboBox()
        self.segment_combo.addItems(["阈值分割", "边缘检测", "区域分割"])

        self.mask_edit_combo = QComboBox()
        self.mask_edit_combo.addItems(["反转 Mask", "去除小区域"])

        self.bg_edit_combo = QComboBox()
        self.bg_edit_combo.addItems(["替换背景", "模糊背景", "绘制蒙版"])

        self.display_mode_combo = QComboBox()
        self.display_mode_combo.addItems(["分割结果显示", "Mask图"])
        self.display_mode_combo.currentIndexChanged.connect(self.update_result_label)

        # 基本操作行
        top_btn_layout = QHBoxLayout()
        for btn in [self.btn_open,self.btn_edit, self.btn_save]:
            top_btn_layout.addWidget(btn)
        top_btn_layout.addWidget(QLabel("分割方式："))
        top_btn_layout.addWidget(self.segment_combo)
        top_btn_layout.addWidget(self.btn_mask)

        # 编辑操作行
        edit_btn_layout = QHBoxLayout()
        edit_btn_layout.addWidget(QLabel("显示方式："))
        edit_btn_layout.addWidget(self.display_mode_combo)
        edit_btn_layout.addWidget(QLabel("Mask 编辑："))
        edit_btn_layout.addWidget(self.mask_edit_combo)
        edit_btn_layout.addWidget(self.btn_edit_mask)
        edit_btn_layout.addWidget(QLabel("目标编辑："))
        edit_btn_layout.addWidget(self.bg_edit_combo)
        edit_btn_layout.addWidget(self.btn_edit_bg)
        edit_btn_layout.addWidget(self.btn_confirm_draw)
        self.btn_confirm_draw.setEnabled(False)  # 初始不可点

        morph_layout = QHBoxLayout()
        morph_layout.addWidget(self.morph_label)
        morph_layout.addWidget(self.morph_combo)
        morph_layout.addWidget(QLabel("核大小"))
        morph_layout.addWidget(self.kernel_slider)
        morph_layout.addWidget(self.btn_morph)

        layout = QVBoxLayout()
        layout.addLayout(top_btn_layout)
        layout.addLayout(edit_btn_layout)
        layout.addWidget(splitter)
        layout.addLayout(morph_layout)

        self.setLayout(layout)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            self.image = file_io.read_image(file_path)
            self.image_label.setPixmap(cv2_to_qpixmap(self.image))

    def edit_target(self):
        if self.image is None or self.latest_binary is None:
            QMessageBox.warning(self, "提示", "请先生成Mask")
            return
        binary = self.latest_binary.copy()
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
            ksize += 1  # 保证奇数核

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))

        if method == "腐蚀":
            processed = cv2.erode(binary, kernel)
        elif method == "膨胀":
            processed = cv2.dilate(binary, kernel)
        elif method == "开运算":
            processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        elif method == "闭运算":
            processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        else:
            processed = binary

        self.latest_binary = processed  # 👉 用于后续编辑
        self.mask = recognition_edit.create_mask_image(processed)
        self.mask_label.setPixmap(cv2_to_qpixmap(self.mask))

    def generate_mask(self):
        if self.image is None:
            QMessageBox.warning(self, "提示", "请先加载图像")
            return

        method = self.segment_combo.currentText()
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        if method == "阈值分割":
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            binary = recognition_edit.refine_segmentation(binary)

        elif method == "边缘检测":
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad = cv2.magnitude(grad_x, grad_y)
            grad = cv2.convertScaleAbs(grad)
            _, binary = cv2.threshold(grad, 50, 255, cv2.THRESH_BINARY)

        elif method == "区域分割":
            _, temp = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            num_labels, labels = cv2.connectedComponents(temp)
            # 保留最大非背景区域
            areas = [np.sum(labels == i) for i in range(1, num_labels)]
            if areas:
                max_label = np.argmax(areas) + 1
                binary = np.uint8((labels == max_label) * 255)
            else:
                binary = temp

        else:
            QMessageBox.warning(self, "错误", "未知分割方法")
            return

        # 保存并显示 mask
        self.latest_binary = binary
        self.mask = recognition_edit.create_mask_image(binary)
        self.mask_label.setPixmap(cv2_to_qpixmap(self.mask))
        self.update_result_label()

    def edit_mask(self):
        if self.latest_binary is None:
            QMessageBox.warning(self, "提示", "请先生成Mask图")
            return

        method = self.mask_edit_combo.currentText()
        binary = self.latest_binary.copy()

        if method == "反转 Mask":
            binary = cv2.bitwise_not(binary)

        elif method == "去除小区域":
            # 连通域分析：保留最大区域
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
            sizes = stats[1:, cv2.CC_STAT_AREA]
            if len(sizes) == 0:
                QMessageBox.information(self, "提示", "未检测到目标区域")
                return
            max_label = 1 + np.argmax(sizes)
            binary = np.uint8((labels == max_label) * 255)

        else:
            QMessageBox.warning(self, "错误", "未知编辑方法")
            return

        # 更新 latest_binary 和 mask 显示
        self.latest_binary = binary
        self.mask = recognition_edit.create_mask_image(binary)
        self.mask_label.setPixmap(cv2_to_qpixmap(self.mask))
        self.update_result_label()

    def edit_background(self):
        try:
            if self.latest_binary is None:
                QMessageBox.warning(self, "提示", "请先生成Mask图")
                return

            method = self.bg_edit_combo.currentText()
            binary = self.latest_binary.copy()

            if method == "替换背景":
                file_path, _ = QFileDialog.getOpenFileName(self, "选择背景图片", "", "Images (*.png *.jpg *.bmp)")
                if not file_path:
                    return  # 用户取消选择

                # 使用背景图替换背景
                bg_img = read_image(file_path)  # 选择或加载背景图
                bg_img_resized = cv2.resize(bg_img, (self.image.shape[1], self.image.shape[0]))
                fg_mask = cv2.bitwise_and(self.image, self.image, mask=binary)
                bg_mask = cv2.bitwise_and(bg_img_resized, bg_img_resized, mask=cv2.bitwise_not(binary))
                result_img = cv2.add(fg_mask, bg_mask)

                self.result = result_img
                self.result_label.setPixmap(cv2_to_qpixmap(self.result))

            elif method == "模糊背景":
                # 使用当前显示图像为基础（即 result 如果有，否则 image）
                source_img = self.result.copy() if self.result is not None else self.image.copy()

                # 对背景区域进行高斯模糊
                blurred = cv2.GaussianBlur(source_img, (21, 21), 0)
                fg_mask = cv2.bitwise_and(source_img, source_img, mask=binary)
                bg_mask = cv2.bitwise_and(blurred, blurred, mask=cv2.bitwise_not(binary))
                result_img = cv2.add(fg_mask, bg_mask)

                self.result = result_img
                self.result_label.setPixmap(cv2_to_qpixmap(self.result))

            elif method == "绘制蒙版":
                if self.image is None:
                    QMessageBox.warning(self, "错误", "请先加载图像")
                    return
                if self.latest_binary is None:
                    QMessageBox.warning(self, "错误", "请先生成初始抠图或Mask")
                    return

                self.drawing_mode = True
                self.btn_confirm_draw.setEnabled(True)

                # 初始化手绘蒙版
                h, w = self.image.shape[:2]
                self.drawing_mask = self.latest_binary.copy()  # 在已有mask上继续绘制
                self.last_x = self.last_y = None

                # 更新显示（可以让用户看到原图并准备绘制）
                self.update_result_label()
                QMessageBox.information(self, "提示", "请在图像上绘制前景区域，再点击“确认绘制”完成。")


            else:
                QMessageBox.warning(self, "错误", "未知背景编辑方法")
                return

        except Exception as e:
            print("edit_background 函数出错：", e)

    # 鼠标事件处理⬇
    def map_to_image_coords(self, x, y):
        """将 QLabel 上的坐标映射到图像坐标"""
        label_width = self.result_label.width()
        label_height = self.result_label.height()
        img_height, img_width = self.image.shape[:2]

        x_ratio = img_width / label_width
        y_ratio = img_height / label_height

        mapped_x = int(x * x_ratio)
        mapped_y = int(y * y_ratio)
        return mapped_x, mapped_y

    def mouse_press_event(self, event):
        if self.drawing_mode and event.button() == Qt.LeftButton:
            x, y = self.map_to_image_coords(event.pos().x(), event.pos().y())
            self.last_x, self.last_y = x, y

    def mouse_move_event(self, event):
        if self.drawing_mode and event.buttons() & Qt.LeftButton:
            x, y = self.map_to_image_coords(event.pos().x(), event.pos().y())
            if self.last_x is not None and self.last_y is not None:
                # 在 mask 上画线
                cv2.line(self.drawing_mask, (self.last_x, self.last_y), (x, y), 255, 10)
                self.last_x, self.last_y = x, y

                # 可视化叠加绘制结果
                overlay = self.image.copy()
                overlay[self.drawing_mask == 255] = [0, 255, 0]  # 绿色表示前景
                self.result_label.setPixmap(cv2_to_qpixmap(overlay))

    def update_result_label(self):
        if self.image is None:
            return

        preview = self.image.copy()

        if self.display_mode_combo.currentText() == "分割结果显示":
            if self.latest_binary is not None:
                red_overlay = np.zeros_like(preview)
                red_overlay[:, :, 2] = self.latest_binary  # 红色通道显示mask
                preview = cv2.addWeighted(preview, 1.0, red_overlay, 0.5, 0)
            self.mask_label.setPixmap(cv2_to_qpixmap(preview))
        elif self.display_mode_combo.currentText() == "Mask图":
            if self.latest_binary is not None:
                preview = recognition_edit.create_mask_image(self.latest_binary)
            self.mask_label.setPixmap(cv2_to_qpixmap(preview))

        # 如果在绘制状态，叠加红色 mask 可视化
        if self.drawing_mode and self.drawing_mask is not None:
            red_overlay = np.zeros_like(preview)
            red_overlay[:, :, 2] = self.drawing_mask  # 将 mask 显示为红色通道
            preview = cv2.addWeighted(preview, 1.0, red_overlay, 0.5, 0)

            self.result_label.setPixmap(cv2_to_qpixmap(preview))

    # 鼠标事件处理⬆

    def confirm_drawing(self):
        if self.drawing_mask is None:
            QMessageBox.warning(self, "提示", "没有绘制任何内容")
            return

        self.latest_binary = self.drawing_mask.copy()
        self.mask_label.setPixmap(cv2_to_qpixmap(self.latest_binary))
        self.drawing_mode = False
        self.btn_confirm_draw.setEnabled(False)
        self.result_label.setText("已应用绘制的蒙版")
        self.update_result_label()


