# 图像识别系统 (基于 CPU，PyQt GUI)

## 📦 功能概览

- 图像读取、保存（支持 JPG, PNG, BMP）
- 摄像头图像抓取（可选）
- 图像预处理（形态学腐蚀/膨胀，区域特征测量）
- 图像分割（阈值分割、边缘检测、区域生长）
- 抠图、生成Mask、图像编辑

## 🚀 运行环境

- Python 3.7+
- OpenCV
- NumPy
- Pillow
- PyQt5

```bash
pip install opencv-python numpy pillow PyQt5
```

## ▶️ 启动方式

```bash
python main.py
```

## 📁 项目结构

```
image_recognition_system/
├── core/              # 核心图像处理模块
├── gui/               # 图形界面（PyQt）
├── main.py            # 主程序入口
├── test_images/       # 测试图像目录
└── README.md
```
