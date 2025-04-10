import torch
import platform
import sys
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QRect, QPoint,QSize,QSettings,QThread, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QProgressBar, QFileDialog, 
                            QLabel, QFrame, QSizePolicy, QSlider, QSizeGrip,QMessageBox,QStackedWidget,QGroupBox,QRadioButton,QSpacerItem)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QMouseEvent, QColor,QFont
import cv2
import numpy as np
import os
import time

from inpaint.InpaintManager import InpaintManager

import config

class VideoProcessingThread(QThread):
    """处理视频的线程"""
    finished = pyqtSignal(str, int,int)  # 定义信号，用于通知主线程处理完成

    def __init__(self,parent, video_path, mask_path,mode):
        super().__init__()
        self.video_processor = parent
        self.video_path = video_path
        self.mask_path = mask_path
        self.mode = mode
    
    def update(self,progress):
        # 发射信号
        if progress > 99:
            progress = 99
        elif progress < 1:
            progress = 1

        self.finished.emit(None,-1,progress)
    

    def run(self):
        """线程运行的逻辑"""
        # 记录开始时间
        start = time.time()
        print("VideoProcessingThread....")
        # 模拟耗时操作
        manager = InpaintManager(self.video_path, self.mask_path,mode=self.mode,callback=self.update)
        manager()
        
        # 记录结束时间
        time_cost = int(time.time() - start)
        
        # 发射信号，通知主线程处理完成
        self.finished.emit(manager.video_out_path, time_cost,100)

class VideoFrame(QLabel):
    selection_changed = pyqtSignal(QRect)
    
    def __init__(self, parent=None, video_path=None,mask_path=None):
        super().__init__(parent)
        self.videoProcessor = parent
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            background-color: black;
            border: 2px solid #444;
            border-radius: 4px;
        """)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.selection_rect = QRect()
        self.dragging = False
        self.start_pos = QPoint()
        self.current_frame = None
        self.scale_factor = 1.0
        self.setMinimumSize(500, 800)
        self.resize(500, 800)
        self.video_path = None
        self.video_cap = None
        self.frame_count = 0
        self.fps = 0
        
    def set_frame(self, frame):
        self.current_frame = frame
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
            height, width = frame.shape[:2]
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled_pixmap)
        else:
            # 创建一个纯黑色的 QImage
            black_image = QImage(self.width(), self.height(), QImage.Format_RGB32)
            black_image.fill(Qt.black)
            pixmap = QPixmap.fromImage(black_image)
            self.setPixmap(pixmap)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.pixmap():
            self.dragging = True
            self.start_pos = event.pos()
            self.selection_rect = QRect(self.start_pos, QSize())
            self.update()
    
    def mouseMoveEvent(self, event):
        if self.dragging and self.pixmap():
            # 计算矩形的左上角和右下角坐标
            top_left = QPoint(min(self.start_pos.x(), event.pos().x()), 
                            min(self.start_pos.y(), event.pos().y()))
            bottom_right = QPoint(max(self.start_pos.x(), event.pos().x()), 
                                max(self.start_pos.y(), event.pos().y()))
            self.selection_rect = QRect(top_left, bottom_right)
            self.update()
    
    def mouseReleaseEvent(self, event):
        if self.dragging and self.pixmap():
            self.dragging = False
            if self.selection_rect.width() > 10 and self.selection_rect.height() > 10:
                # 打印矩形框的坐标
                print(f"矩形框坐标: x={self.selection_rect.x()}, y={self.selection_rect.y()}, "
                    f"width={self.selection_rect.width()}, height={self.selection_rect.height()}")

                # 生成蒙版图
                if self.current_frame is not None:
                    original_height, original_width = self.current_frame.shape[:2]
                    display_width = self.pixmap().width()
                    display_height = self.pixmap().height()

                    # 计算缩放比例
                    scale_x = original_width / display_width
                    scale_y = original_height / display_height

                    # 计算偏移量（如果视频帧被居中显示）
                    offset_x = (self.width() - display_width) // 2
                    offset_y = (self.height() - display_height) // 2

                    # 将显示区域的坐标转换为原始视频帧的坐标
                    x = int((self.selection_rect.x() - offset_x) * scale_x)
                    y = int((self.selection_rect.y() - offset_y) * scale_y)
                    w = int(self.selection_rect.width() * scale_x)
                    h = int(self.selection_rect.height() * scale_y)

                    # 确保坐标在视频帧范围内
                    x = max(0, min(x, original_width))
                    y = max(0, min(y, original_height))
                    w = max(0, min(w, original_width - x))
                    h = max(0, min(h, original_height - y))

                    # 创建蒙版图
                    mask = np.zeros((original_height, original_width), dtype=np.uint8)
                    mask[y:y+h, x:x+w] = 255  # 将矩形区域设置为白色

                    # 获取视频文件路径和名称
                    video_path = self.videoProcessor.video_path  # 从父窗口获取视频路径
                    video_dir = os.path.dirname(video_path)
                    video_name = os.path.basename(video_path).split('.')[0]
                    mask_path = os.path.join(video_dir, f"{video_name}_mask.png")
                  
                    # 确保路径使用正斜杠
                    self.save_mask(mask,mask_path)

            else:
                self.selection_rect = QRect()
            self.update()
    
    def paintEvent(self, event):
        super().paintEvent(event)
        if self.selection_rect and self.pixmap():
            painter = QPainter(self)
            painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.SolidLine))
            painter.drawRect(self.selection_rect)
    
    def save_mask(self,mask, mask_path):
        # 确保目录存在
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)

        # 检查权限
        if not os.access(os.path.dirname(mask_path), os.W_OK):
            print(f"没有权限写入路径: {os.path.dirname(mask_path)}")
            return

        # 检查蒙版图数据
        if mask is None or mask.size == 0:
            print("蒙版图为空")
            return

        if mask.dtype != np.uint8:
            print(f"蒙版图数据类型错误: {mask.dtype}")
            return

        # 检查文件名
        if not mask_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"文件名没有有效的扩展名: {mask_path}")
            return

        # 保存蒙版图
        if cv2.imwrite(mask_path, mask):
            self.videoProcessor.mask_path = mask_path
            print(f"蒙版图已成功保存到: {mask_path}")
        else:
            print(f"保存蒙版图失败: {mask_path}")

class VideoProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频去字幕工具")
        self.setGeometry(500, 500, 500, 800) 
        self.settings = QSettings("VideoSubTitleRemover", "dragon")
        self.last_opened_path = self.settings.value("last_opened_path", "")
        self.video_path = None
        self.mask_path = None
        self.processing_thread = None
        self.inpaint_mode = config.InpaintMode.STTN
        
        # 主窗口布局 - 改为左右结构
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # ===== 左侧导航栏 =====
        left_nav = QFrame()
        left_nav.setFrameShape(QFrame.StyledPanel)
        left_nav.setStyleSheet("background-color: #f0f0f0;")
        left_nav.setFixedWidth(200)
        left_nav_layout = QVBoxLayout(left_nav)
        left_nav_layout.setContentsMargins(10, 20, 10, 20)
        left_nav_layout.setSpacing(15)
        
        # 首页按钮
        self.home_btn = QPushButton("首页")
        self.home_btn.setCheckable(True)
        self.home_btn.setChecked(True)
        self.home_btn.setStyleSheet("""
            QPushButton {
                padding: 12px;
                font-size: 20px;
                text-align: left;
                border: none;
                border-radius: 5px;
            }
            QPushButton:checked {
                background-color: #d0d0d0;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        self.home_btn.clicked.connect(self.show_home)
        
        # 设置按钮
        self.settings_btn = QPushButton("设置")
        self.settings_btn.setCheckable(True)
        self.settings_btn.setStyleSheet(self.home_btn.styleSheet())
        self.settings_btn.clicked.connect(self.show_settings)
        
        # 添加按钮和弹簧
        left_nav_layout.addWidget(self.home_btn)
        left_nav_layout.addWidget(self.settings_btn)
        left_nav_layout.addStretch()
        
        # ===== 右侧内容区域 =====
        self.right_stack = QStackedWidget()
        
        # 首页内容
        self.home_widget = QWidget()
        self.setup_home_ui()
        
        # 设置页面
        self.settings_widget = QWidget()
        self.setup_settings_ui()
        
        self.right_stack.addWidget(self.home_widget)
        self.right_stack.addWidget(self.settings_widget)
        
        # 添加到主布局
        main_layout.addWidget(left_nav)
        main_layout.addWidget(self.right_stack)
        
        # 其他初始化
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.update_frames)
        self.is_playing = False
        
        # 视频相关变量
        self.video_cap = None
        self.frame_count = 0
        self.fps = 0
        self.current_frame_pos = 0
        self.selection_area = None
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.update_frames)
        self.original_width = 0
        self.original_height = 0
        self.is_playing = False
        self.video_duration = 0
        self.slider_dragging = False

    def setup_home_ui(self):
        """首页UI - 原有内容移到这里"""
        self.home_widget.setAttribute(Qt.WA_StyledBackground)  # ← 新增
        layout = QVBoxLayout(self.home_widget)
        layout.setContentsMargins(50, 50, 50,10)
        
        # 视频帧区域
        video_area = QHBoxLayout()
        video_area.setContentsMargins(0, 0, 0, 0)
        video_area.setSpacing(20)
        
        # 原始视频
        self.original_video = VideoFrame(parent=self)
        video_area.addWidget(self.original_video)
        
        
        # 处理后视频
        self.processed_video = VideoFrame(parent=self)
        video_area.addWidget(self.processed_video)
        
        layout.addLayout(video_area)

        
        # 进度标签和滑块
        self.progress_label = QLabel("00:00:00 / 00:00:00")
        layout.addWidget(self.progress_label,alignment=Qt.AlignmentFlag.AlignCenter)
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setStyleSheet("""
                QSlider::groove:horizontal {
                    height: 8px;
                    background: #ddd;  /* 未滑过区域颜色 */
                }
                QSlider::sub-page:horizontal {
                    background: #3396ff;  /* 已滑过区域颜色（绿色） */
                }
                QSlider::handle:horizontal {
                    width: 12px;          /* 圆形直径 */
                    height: 18px;         /* 圆形直径 */
                    margin: -4px 0;       /* 垂直居中 */
                    background: #3396FF;  /* 直接填充为边框颜色（无白色背景） */
                    border: none;        /* 移除边框 */
                }
            """)
        self.progress_slider.sliderPressed.connect(self.slider_pressed)
        self.progress_slider.sliderReleased.connect(self.slider_released)
        self.progress_slider.sliderMoved.connect(self.seek_video)
        self.progress_slider.setEnabled(False)
        layout.addWidget(self.progress_slider)

        spacer = QSpacerItem(50, 50, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        layout.addSpacerItem(spacer)

          # 按钮区域
        button_area = QHBoxLayout()
        button_area.setContentsMargins(0, 0, 0, 0)
        
        self.open_btn = QPushButton("打开视频")
        self.open_btn.clicked.connect(self.open_video)
        self.open_btn.setStyleSheet("""
            QPushButton {
                padding: 15px 60px;
                font-size: 20px;
            }
        """)
        button_area.addWidget(self.open_btn)

        self.play_btn = QPushButton("播放")
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setStyleSheet("""
            QPushButton {
                padding: 15px 60px;
                font-size: 20px;
            }
        """)
        button_area.addWidget(self.play_btn)

        self.process_btn = QPushButton("去除字幕")
        self.process_btn.clicked.connect(self.process_video)
        self.process_btn.setStyleSheet("""
            QPushButton {
                padding: 15px 60px;
                font-size: 20px;
            }
        """)
        button_area.addWidget(self.process_btn)
        layout.addLayout(button_area)

        spacer = QSpacerItem(50, 50, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        layout.addSpacerItem(spacer)
        
        # 处理进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                height: 15px;
            }
        """)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.progress_bar)

        #弹簧
        layout.addStretch(1)

        # 显示环境控件 - 现在会固定在底部
        environmentString = f"Python 版本: {platform.python_version()}" \
                f"     torch 版本: {torch.__version__}," \
                f"     是否支持CUDA: {torch.cuda.is_available()}"
        self.environmentLabel = QLabel(environmentString)
        font = QFont()
        font.setPointSize(6)
        self.environmentLabel.setFont(font)

        layout.addWidget(self.environmentLabel)
        
        # 初始化控件状态
        self.play_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        self.progress_slider.setEnabled(False)
    
    def setup_settings_ui(self):
        """设置页面UI"""
        layout = QVBoxLayout(self.settings_widget)
        layout.setContentsMargins(50, 50, 50, 50)
        layout.setSpacing(30)
        
        # 算法选择组
        algo_group = QGroupBox("修复算法选择")
        algo_group.setStyleSheet("""
            QGroupBox {
                font-size: 18px;
                border: 1px solid #ccc;
                border-radius: 5px;
                margin-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 20px;
            }
        """)
        algo_layout = QVBoxLayout()
        
        self.radio_sttn = QRadioButton("Sttn模型")
        self.radio_propainter = QRadioButton("ProPainter模型")
        self.radio_lama = QRadioButton("Lama模型")
        
        self.radio_sttn.setChecked(True)  # 默认选择sttn
        self.set_inpaint_mode(config.InpaintMode.STTN)
        
        # 添加单选按钮
        algo_layout.addWidget(self.radio_sttn)
        algo_layout.addWidget(self.radio_propainter)
        algo_layout.addWidget(self.radio_lama)
        algo_layout.addSpacing(15)
        
        # 算法说明
        algo_desc = QLabel(
            "Sttn: 适合处理动态字幕\n"
            "Propainter: 适合处理动态字幕\n"
            "Lama: 适合静态字幕和复杂背景修复（推荐图片处理）"
        )
        algo_desc.setStyleSheet("color: #666; font-size: 16px;")
        algo_layout.addWidget(algo_desc)
        
        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group)
        
        # 连接信号
        self.radio_sttn.clicked.connect(
            lambda: self.set_inpaint_mode(config.InpaintMode.STTN))

        self.radio_propainter.clicked.connect(
            lambda: self.set_inpaint_mode(config.InpaintMode.PROPAINTER))

        self.radio_lama.clicked.connect(
            lambda: self.set_inpaint_mode(config.InpaintMode.LAMA))
        
        layout.addStretch()

    def set_inpaint_mode(self, mode):
        """设置修复算法模式"""
        self.inpaint_mode = mode
        print(f"算法模式已切换至: {mode}")

    def show_home(self):
        """显示首页"""
        self.home_btn.setChecked(True)
        self.settings_btn.setChecked(False)
        self.right_stack.setCurrentIndex(0)

    def show_settings(self):
        """显示设置页"""
        self.home_btn.setChecked(False)
        self.settings_btn.setChecked(True)
        self.right_stack.setCurrentIndex(1)

    def resizeEvent(self, event):
            super().resizeEvent(event)
            # 在窗口大小变化时执行的自定义逻辑
            self.on_window_resized()

    def on_window_resized(self):
        # 更新视频帧的显示大小
        if self.original_video.current_frame is not None:
            self.original_video.set_frame(self.original_video.current_frame)
        if self.processed_video.current_frame is not None:
            self.processed_video.set_frame(self.processed_video.current_frame)

    def slider_pressed(self):
        self.slider_dragging = True
        if self.is_playing:
            self.pause_video()

    def slider_released(self):
        self.slider_dragging = False

    def toggle_play(self):
        """切换播放状态"""
        if self.is_playing:
            self.pause_video()
        else:
            self.play_video()

    def play_video(self):
        """播放视频"""
        if self.video_cap and not self.is_playing:
            self.is_playing = True
            self.play_btn.setText("暂停")
            self.play_timer.start(int(1000 / self.fps))

    def pause_video(self):
        """暂停视频"""
        if self.is_playing:
            self.is_playing = False
            self.play_btn.setText("播放")
            self.play_timer.stop()

    def seek_video(self, position):
        """跳转到指定位置"""
        if self.video_cap and self.slider_dragging:
            frame_pos = int(position * self.frame_count / 100)
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            self.current_frame_pos = frame_pos
            ret, frame = self.video_cap.read()
            if ret:
                self.display_frame(frame, self.original_video)
                if self.processed_video.video_path and self.processed_video.video_cap:
                    self.processed_video.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                    ret, processed_frame = self.processed_video.video_cap.read()
                    if ret:
                        self.display_frame(processed_frame, self.processed_video)
            self.update_progress()

    def update_progress(self):
        """更新进度显示"""
        if self.video_cap and not self.slider_dragging:
            # 更新进度滑块
            progress = int(self.current_frame_pos * 100 / self.frame_count)
            if self.current_frame_pos == self.frame_count -1:
                progress = 100

            self.progress_slider.setValue(progress)

         # 更新时间显示
        current_time = self.current_frame_pos / self.fps
        total_time = self.frame_count / self.fps
        
        current_str = self.format_time(current_time)
        total_str = self.format_time(total_time)
        
        self.progress_label.setText(f"{current_str} / {total_str}")

    def update_process(self,progress):
        self.progress_bar.setValue(progress)

    def format_time(self, seconds):
        """格式化时间显示"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def process_video(self):

        print("开始处理视频")
        
        # 禁用控件
        self.set_controls_enabled(False)
        self.progress_bar.setValue(0)
        
        self.subtitle_processing()

    def releaseResource(self):
        if self.video_cap:
            self.video_cap.release() 
            self.video_cap = None
        
        if self.processed_video.video_cap:
            print("releaseResource,processed_video")
            self.processed_video.video_cap.release() 
            self.processed_video.video_cap = None
            self.processed_video.set_frame(None)
    

    def open_video(self):
        """打开视频文件"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", self.last_opened_path, 
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)"
        )
        
        if filename:
            self.video_path = filename  # 保存视频路径
            self.original_video.video_path = filename  # 更新 original_video 的 video_path
            self.processed_video.video_path = filename  # 更新 processed_video 的 video_path
            self.last_opened_path = os.path.dirname(filename)  # 更新上次打开的路径
            self.settings.setValue("last_opened_path", self.last_opened_path)  # 保存路径到 QSettings

            self.releaseResource()

            self.video_cap = cv2.VideoCapture(filename)
            if self.video_cap.isOpened():
                self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
                self.current_frame_pos = 0
                
                # 读取第一帧
                ret, frame = self.video_cap.read()
                if ret:
                    # 保存原始视频尺寸
                    self.original_height, self.original_width = frame.shape[:2]
                    self.display_frame(frame, self.original_video)
                
                # 启用控件
                self.play_btn.setEnabled(True)
                self.process_btn.setEnabled(True)
                self.progress_slider.setEnabled(True)
                
                # 初始化进度条
                self.progress_slider.setRange(0, 100)
                self.update_progress()
            else:
                print("无法打开视频文件")

    def handle_selection_changed(self, rect):
        """处理选择的区域变化"""
        self.selection_area = rect
        if self.original_video.current_frame is not None and rect:
            frame = self.original_video.current_frame.copy()
            processed_frame = self.process_frame(frame)
            self.display_frame(processed_frame, self.processed_video)

    def process_frame(self, frame):
        """处理单帧图像 - 这里简单地将选中区域模糊化"""
        if self.selection_area:
            x, y, w, h = self.selection_area.x(), self.selection_area.y(), self.selection_area.width(), self.selection_area.height()
            roi = frame[y:y+h, x:x+w]
            blurred = cv2.GaussianBlur(roi, (25, 25), 0)
            frame[y:y+h, x:x+w] = blurred
        return frame

    def update_frames(self):
        """更新视频帧显示"""
        if self.video_cap and self.video_cap.isOpened() and self.is_playing:
            ret, frame = self.video_cap.read()
            self.current_frame_pos += 1
            
            if ret:
                self.display_frame(frame, self.original_video)
                if self.processed_video.video_path and self.processed_video.video_cap:
                    self.processed_video.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_pos)
                    ret, processed_frame = self.processed_video.video_cap.read()
                    if ret:
                        self.display_frame(processed_frame, self.processed_video)
                self.update_progress()
            else:
                # 视频结束，回到开头
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame_pos = 0
                self.pause_video()
    
    def display_frame(self, frame, display_widget):
        """显示视频帧"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # 缩放图像以适应标签大小，同时保持宽高比
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            display_widget.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        display_widget.setPixmap(scaled_pixmap)
        display_widget.current_frame = frame

    def set_controls_enabled(self, enabled):
        """设置控件启用状态"""
        self.open_btn.setEnabled(enabled)
        self.play_btn.setEnabled(enabled)
        self.process_btn.setEnabled(enabled)
        self.progress_slider.setEnabled(enabled)

    def subtitle_processing(self):
        """启动视频处理"""
        if self.video_path and self.mask_path:
            # 禁用控件
            self.set_controls_enabled(False)
            
            # 重置进度条
            self.progress_bar.setValue(0)
            self.pause_video()
            
            # 启动处理线程
            self.processing_thread = VideoProcessingThread(self,self.video_path,self.mask_path,self.inpaint_mode)
            self.processing_thread.finished.connect(self.processing_finished)
            self.processing_thread.start()
        else : 
            self.set_controls_enabled(True)
            QMessageBox.information(self, "信息", "请选择文件或选择去字幕区域")

    def processing_finished(self, output_path, time_cost,progress):
        if not output_path and time_cost == -1:
            self.progress_bar.setValue(progress)
            return

        self.progress_bar.setValue(100)
        self.set_controls_enabled(True)
        self.original_video.selection_rect = QRect()
        self.original_video.update()
        QMessageBox.information(self, "处理完成", f"视频已生成到 {output_path}\n耗时: {time_cost} 秒")
        self.processed_video.video_path = output_path
        self.processed_video.video_cap = cv2.VideoCapture(output_path)
        self.processed_video.frame_count = int(self.processed_video.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.processed_video.fps = self.processed_video.video_cap.get(cv2.CAP_PROP_FPS)
        self.processed_video.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_pos)
        ret, frame = self.processed_video.video_cap.read()
        if ret:
            self.progress_bar.setValue(0)
            self.display_frame(frame, self.processed_video)


if __name__ == "__main__":
    # 打印环境信息
    print("Python 版本:", platform.python_version()) 
    print("torch 版本:", torch.__version__) 
    print("是否支持CUDA: ", torch.cuda.is_available())
    print("环境检测完毕")
    
    app = QApplication(sys.argv)
    
    # 设置全局字体
    font = QFont("Microsoft YaHei", 10) 
    app.setFont(font)
    font.setBold(True)
    
    window = VideoProcessor()
    window.show()
    sys.exit(app.exec_())