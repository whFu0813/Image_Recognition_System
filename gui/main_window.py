import sys

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QComboBox, QSlider,
    QFileDialog, QHBoxLayout, QMessageBox, QScrollArea, QSplitter, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer
from core import file_io, segmentation, recognition_edit
from core.file_io import read_image
from gui.sticker_window import StickerWindow
from gui.ui_utils import cv2_to_qpixmap
from gui.draw_mask_window import DrawMaskWindow

import cv2

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像识别系统")
        self.resize(1600, 1000)

        self.image = None
        self.cap = None
        self.timer = None
        self.mask = None
        self.result = None
        self.latest_binary = None  # 用于编辑阶段的mask/分割结果
        self.drawing_mode = False  # 绘制蒙版模式
        self.drawing_mask = None
        self.last_x, self.last_y = None, None  # 绘制中存储上次鼠标坐标
        self.draw_win = None
        self.sticker_win = None

        # 为实现贴纸添加的拖拽
        self.dragging = False
        self.drag_offset = (0, 0)
        self.foreground = None  # 抠图目标区域
        self.fg_mask = None  # 掩码
        self.fg_position = (0, 0)

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

        # mask 区域下方添加显示方式选择
        mask_area_widget = QWidget()
        mask_area_layout = QVBoxLayout()
        mask_area_layout.setContentsMargins(0, 0, 0, 0)
        mask_area_layout.setSpacing(5)

        # 原有的 mask_label 封装
        mask_area_layout.addWidget(scroll_mask)

        scroll_result = QScrollArea()
        scroll_result.setWidgetResizable(True)
        scroll_result.setWidget(self.result_label)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll_image)
        splitter.addWidget(mask_area_widget)
        splitter.addWidget(scroll_result)
        scroll_image.setMinimumWidth(530)
        mask_area_widget.setMinimumWidth(540)
        scroll_result.setMinimumWidth(530)

        self.btn_open = QPushButton("打开图像")
        self.btn_open.clicked.connect(self.open_image)
        self.btn_camera = QPushButton("打开摄像头")
        self.btn_camera.clicked.connect(self.open_camera)
        self.btn_capture = QPushButton("抓拍图像")
        self.btn_capture.clicked.connect(self.capture_frame)
        self.btn_capture.setEnabled(False)  # 打开摄像头后才可抓拍
        self.btn_edit = QPushButton("应用目标抠图")
        self.btn_edit.clicked.connect(self.edit_target)
        self.export_transparent_btn = QPushButton("导出透明背景图像")
        self.export_transparent_btn.clicked.connect(self.export_transparent_result)
        self.btn_save = QPushButton("保存结果")
        self.btn_save.clicked.connect(self.save_result)
        self.btn_morph = QPushButton("应用形态学处理")
        self.btn_morph.clicked.connect(self.apply_morphology)
        self.btn_mask = QPushButton("生成 Mask")
        self.btn_mask.clicked.connect(self.generate_mask)
        self.btn_feature = QPushButton("特征测量")
        self.btn_feature.clicked.connect(self.measure_features)
        self.btn_edit_mask = QPushButton("Mask处理")
        self.btn_edit_mask.clicked.connect(self.edit_mask)
        self.btn_edit_bg = QPushButton("背景处理")
        self.btn_edit_bg.clicked.connect(self.edit_background)

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
        self.bg_edit_combo.addItems(["替换背景", "模糊背景", "绘制蒙版", "贴纸拖动"])

        self.display_mode_combo = QComboBox()
        self.display_mode_combo.addItems(["分割结果显示", "Mask图"])
        self.display_mode_combo.currentIndexChanged.connect(self.update_result_label)

        # 基本操作行
        top_btn_layout = QHBoxLayout()
        top_btn_layout.addWidget(self.btn_camera)
        top_btn_layout.addWidget(self.btn_capture)
        top_btn_layout.addWidget(self.btn_open)
        top_btn_layout.addWidget(self.btn_edit)
        top_btn_layout.addWidget(self.export_transparent_btn)
        top_btn_layout.addWidget(self.btn_save)
        top_btn_layout.addWidget(QLabel("分割方式："))
        top_btn_layout.addWidget(self.segment_combo)
        top_btn_layout.addWidget(self.btn_mask)
        top_btn_layout.addWidget(self.btn_feature)

        # 编辑操作行
        edit_btn_layout = QHBoxLayout()
        edit_btn_layout.addWidget(QLabel("Mask 编辑："))
        edit_btn_layout.addWidget(self.mask_edit_combo)
        edit_btn_layout.addWidget(self.btn_edit_mask)
        edit_btn_layout.addWidget(QLabel("目标编辑："))
        edit_btn_layout.addWidget(self.bg_edit_combo)
        edit_btn_layout.addWidget(self.btn_edit_bg)
        #edit_btn_layout.addStretch()   #让布局向左靠齐，空出右侧

        # 显示方式选择器
        display_mode_layout = QHBoxLayout()
        display_mode_layout.setContentsMargins(0, 0, 0, 0)

        display_mode_label = QLabel("显示方式：")
        display_mode_layout.addWidget(display_mode_label)
        display_mode_layout.addWidget(self.display_mode_combo)
        display_mode_layout.addStretch()

        mask_area_layout.addLayout(display_mode_layout)
        mask_area_widget.setLayout(mask_area_layout)

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

    def open_camera(self):
        if self.cap is not None:
            # 如果已打开，则关闭摄像头
            self.cap.release()
            self.cap = None
            if self.timer:
                self.timer.stop()
            self.btn_camera.setText("打开摄像头")
            self.btn_capture.setEnabled(False)
            return

        # 打开摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开摄像头")
            self.cap = None
            return

        self.btn_camera.setText("关闭摄像头")
        self.btn_capture.setEnabled(True)

        # 启动定时器预览图像
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera_frame)
        self.timer.start(30)

    def update_camera_frame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        self.image_label.setPixmap(cv2_to_qpixmap(frame))

    def capture_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            QMessageBox.warning(self, "错误", "抓拍失败")
            return

        # 停止摄像头并清理资源
        self.timer.stop()
        self.cap.release()
        self.cap = None
        self.btn_camera.setText("打开摄像头")
        self.btn_capture.setEnabled(False)

        # 保存并显示图像
        self.image = frame.copy()
        self.image_label.setPixmap(cv2_to_qpixmap(self.image))
        QMessageBox.information(self, "提示", "图像抓拍成功，可进行处理")

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

    def export_transparent_result(self):
        if self.image is None or self.latest_binary is None:
            QMessageBox.warning(self, "提示", "请先生成Mask图")
            return

        # 构建透明图像
        b, g, r = cv2.split(self.image)
        alpha = self.latest_binary.copy()
        rgba = cv2.merge((b, g, r, alpha))  # 形成 BGRA 图像

        # 保存为 PNG
        file_path, _ = QFileDialog.getSaveFileName(self, "保存透明背景图像", "", "PNG Files (*.png)")
        if file_path:
            if not file_path.lower().endswith(".png"):
                file_path += ".png"
            cv2.imwrite(file_path, rgba)
            QMessageBox.information(self, "保存成功", "透明背景图像已保存为PNG")

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

    def measure_features(self):
        if self.latest_binary is None:
            QMessageBox.warning(self, "提示", "请先生成或处理二值图像")
            return

        results = recognition_edit.measure_features(self.latest_binary)

        if not results:
            QMessageBox.information(self, "提示", "未检测到目标区域")
            return

        # 将结果格式化为字符串
        text = "\n".join([
            f"面积: {r['area']}, 周长: {r['perimeter']}, 重心: ({r['centroid'][0]:.2f}, {r['centroid'][1]:.2f})"
            for r in results
        ])
        QMessageBox.information(self, "特征测量结果", text)

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

                # 弹出绘制窗口
                def receive_updated_mask(new_mask):
                    self.latest_binary = new_mask
                    self.mask = recognition_edit.create_mask_image(new_mask)
                    self.mask_label.setPixmap(cv2_to_qpixmap(self.mask))
                    self.update_result_label()
                    QMessageBox.information(self, "提示", "蒙版已更新")

                self.draw_win = DrawMaskWindow(self.image, self.latest_binary, receive_updated_mask)
                self.draw_win.setAttribute(Qt.WA_DeleteOnClose)  # 关闭后自动删除
                self.draw_win.show()

            elif method == "贴纸拖动":
                if self.image is None or self.latest_binary is None:
                    QMessageBox.warning(self, "提示", "请先加载图像并生成Mask")
                    return

                # 回调函数，接收贴图处理后的结果图像
                def receive_sticker_result(new_image):
                    self.result = new_image
                    self.result_label.setPixmap(cv2_to_qpixmap(self.result))
                    QMessageBox.information(self, "提示", "贴图操作已完成")

                self.sticker_win = StickerWindow(self.image, self.latest_binary, receive_sticker_result)
                self.sticker_win.setAttribute(Qt.WA_DeleteOnClose)
                self.sticker_win.show()


            else:
                QMessageBox.warning(self, "错误", "未知背景编辑方法")
                return

        except Exception as e:
            print("edit_background 函数出错：", e)

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




