import os
import numpy as np
from PyQt5.QtCore import Qt, QRect, QPoint, QSize, pyqtSignal
from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush
from  inpaint.utils.utils import save_image_with_chinese_path

class VideoFrame(QLabel):
    selection_changed = pyqtSignal(QRect)

    def __init__(self, parent=None, screenWidth = 1920, screenHeight = 1080):
        
        super().__init__()
        self.videoProcessor = parent
        self.video_cap = None
        self.video_path = None

        self.initWidth = screenWidth * 0.2
        self.initHeight = screenHeight * 0.4

        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            background-color: black;
            border: 2px solid #444;
            border-radius: 4px;
        """)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.selection_rects = []  # 存储原始内容坐标的QRect
        self.dragging = False
        self.resizing = False
        self.current_rect_idx = -1
        self.start_pos = QPoint()
        self.resize_start_rect = None
        self.current_frame = None
        self.setMinimumSize(int(self.initWidth), int(self.initHeight))
        self.resize(int(self.initWidth), int(self.initHeight))
        self.add_button_size = QSize(16, 16)
        self.close_button_size = QSize(16, 16)
        self.resize_button_size = QSize(16, 16)
        self.max_rects = 3
        self.drawAble = False  # 是否可以设置选区
        self.offset_x = 0
        self.offset_y = 0
        self.scale = 1.0
        self.frame_width = 1
        self.frame_height = 1
        self.enabled = True  # 该控件是否相应鼠标事件

    def set_frame(self, frame):
        self.current_frame = frame
        if frame is not None:
            self.frame_height, self.frame_width = frame.shape[:2]
            bytes_per_line = 3 * self.frame_width
            q_img = QImage(frame.data, self.frame_width, self.frame_height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled_pixmap)
            # 计算缩放和偏移
            scale_x = self.width() / self.frame_width
            scale_y = self.height() / self.frame_height
            self.scale = min(scale_x, scale_y)
            scaled_width = int(self.frame_width * self.scale)
            scaled_height = int(self.frame_height * self.scale)
            self.offset_x = (self.width() - scaled_width) // 2
            self.offset_y = (self.height() - scaled_height) // 2
            
            # 默认添加一个矩形
            if len(self.selection_rects) == 0:
                w = self.frame_width // 2
                h = self.frame_height // 10
                x = 100
                y = self.frame_height // 2

                self.selection_rects.append(QRect(x, y, w, h))
                print(f"添加默认矩形: {self.selection_rects[0]}")
            self.update()
        else:
            black_image = QImage(self.width(), self.height(), QImage.Format_RGB32)
            black_image.fill(Qt.black)
            pixmap = QPixmap.fromImage(black_image)
            self.setPixmap(pixmap)
            self.update()

    def clear_selections(self):
        self.selection_rects.clear()

    def setEnabled(self, enabled):
        # 只改变交互，不调用父类的 setEnabled，避免 QLabel 自动变灰
        self.enabled = enabled
        self.repaint()  # 触发重绘

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_frame is not None:
            self.set_frame(self.current_frame)
    
    def setDrawable(self, drawable):
        """设置是否可以设置选区"""
        self.drawAble = drawable
        if not drawable:
            self.selection_rects.clear()
            self.update()

    def map_to_display(self, rect):
        """内容坐标->显示坐标"""
        return QRect(
            int(rect.left() * self.scale) + self.offset_x,
            int(rect.top() * self.scale) + self.offset_y,
            int(rect.width() * self.scale),
            int(rect.height() * self.scale)
        )

    def map_to_content(self, point):
        """显示坐标->内容坐标"""
        x = int((point.x() - self.offset_x) / self.scale)
        y = int((point.y() - self.offset_y) / self.scale)
        return QPoint(x, y)

    def is_rect_overlapping(self, new_rect, ignore_idx=-1):
        for idx, rect in enumerate(self.selection_rects):
            if idx == ignore_idx:
                continue
            if new_rect.intersects(rect):
                return True
        return False

    def is_rect_within_bounds(self, rect):
        return (0 <= rect.left() and 0 <= rect.top() and
                rect.right() < self.frame_width and rect.bottom() < self.frame_height and
                rect.width() > 0 and rect.height() > 0)

    def mousePressEvent(self, event):
        if not self.enabled or not self.drawAble or self.current_frame is None or self.isEnabled() == False:
            return
        pos = event.pos()
        for idx, rect in enumerate(self.selection_rects):
            display_rect = self.map_to_display(rect)
            # 右下角缩放
            resize_btn = QRect(display_rect.right() - self.resize_button_size.width() + 1,
                               display_rect.bottom() - self.resize_button_size.height() + 1,
                               self.resize_button_size.width(), self.resize_button_size.height())
            if resize_btn.contains(pos):
                self.resizing = True
                self.current_rect_idx = idx
                self.start_pos = pos
                self.resize_start_rect = QRect(rect)
                return
            # 右上角删除
            close_btn = QRect(display_rect.right() - self.close_button_size.width() + 1,
                              display_rect.top(),
                              self.close_button_size.width(), self.close_button_size.height())
            if close_btn.contains(pos) and len(self.selection_rects) > 1:
                self.selection_rects.pop(idx)
                self.update()
                return
            # 左上角添加
            add_btn = QRect(display_rect.left(), display_rect.top(),
                            self.add_button_size.width(), self.add_button_size.height())
            if add_btn.contains(pos) and len(self.selection_rects) < self.max_rects:
                # 新矩形默认不重叠，找一个合适位置
                for try_y in range(0, self.frame_height, 30):
                    for try_x in range(0, self.frame_width, 30):
                        w = self.frame_width // 2
                        h = self.frame_height // 10
                        new_rect = QRect(try_x, try_y, w, h)
                        if (not self.is_rect_overlapping(new_rect)
                            and self.is_rect_within_bounds(new_rect)):
                            self.selection_rects.append(new_rect)
                            self.update()
                            return
                return
            # 拖动区域
            if display_rect.contains(pos):
                self.dragging = True
                self.current_rect_idx = idx
                self.start_pos = pos
                self.drag_start_rect = QRect(rect)
                return

    def mouseMoveEvent(self, event):
        if self.dragging and self.current_rect_idx != -1:
            delta = self.map_to_content(event.pos()) - self.map_to_content(self.start_pos)
            new_rect = QRect(self.drag_start_rect)
            new_rect.moveTo(self.drag_start_rect.topLeft() + delta)
            # 保证在内容区域内
            if new_rect.left() < 0:
                new_rect.moveLeft(0)
            if new_rect.top() < 0:
                new_rect.moveTop(0)
            if new_rect.right() >= self.frame_width:
                new_rect.moveRight(self.frame_width - 1)
            if new_rect.bottom() >= self.frame_height:
                new_rect.moveBottom(self.frame_height - 1)
            # 不重叠
            if not self.is_rect_overlapping(new_rect, self.current_rect_idx) and self.is_rect_within_bounds(new_rect):
                self.selection_rects[self.current_rect_idx] = new_rect
                self.update()
        elif self.resizing and self.current_rect_idx != -1:
            start = self.map_to_content(self.start_pos)
            now = self.map_to_content(event.pos())
            diff = now - start
            new_rect = QRect(self.resize_start_rect)
            new_width = max(20, new_rect.width() + diff.x())
            new_height = max(20, new_rect.height() + diff.y())
            new_rect.setSize(QSize(new_width, new_height))
            # 保证在内容区域内
            if new_rect.right() >= self.frame_width:
                new_rect.setRight(self.frame_width - 1)
            if new_rect.bottom() >= self.frame_height:
                new_rect.setBottom(self.frame_height - 1)
            # 不重叠
            if not self.is_rect_overlapping(new_rect, self.current_rect_idx) and self.is_rect_within_bounds(new_rect):
                self.selection_rects[self.current_rect_idx] = new_rect
                self.update()

    def mouseReleaseEvent(self, event):
        self.dragging = False
        self.resizing = False
        self.current_rect_idx = -1

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.current_frame is None or not self.drawAble:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        for idx, rect in enumerate(self.selection_rects):
            display_rect = self.map_to_display(rect)
            # 主体
            painter.setPen(QPen(QColor(200,200,200), 1, Qt.SolidLine))
            painter.setBrush(QBrush(QColor(255,255,255, 100)))
            painter.drawRect(display_rect)
            # 添加按钮
            if len(self.selection_rects) < self.max_rects:
                add_btn = QRect(display_rect.left(), display_rect.top(),
                                self.add_button_size.width(), self.add_button_size.height())
                painter.setBrush(QBrush(QColor(0, 200, 0, 255)))
                painter.drawRect(add_btn)
                painter.drawText(add_btn, Qt.AlignCenter, "+")
            # 删除按钮
            if len(self.selection_rects) > 1:
                close_btn = QRect(display_rect.right() - self.close_button_size.width() + 1,
                                  display_rect.top(),
                                  self.close_button_size.width(), self.close_button_size.height())
                painter.setBrush(QBrush(QColor(200, 0, 0, 255)))
                painter.drawRect(close_btn)
                painter.drawText(close_btn, Qt.AlignCenter, "x")
            # 缩放按钮
            resize_btn = QRect(display_rect.right() - self.resize_button_size.width() + 1,
                               display_rect.bottom() - self.resize_button_size.height() + 1,
                               self.resize_button_size.width(), self.resize_button_size.height())
            painter.setBrush(QBrush(QColor(255 ,255, 255, 255)))
            painter.drawRect(resize_btn)
            painter.drawText(resize_btn, Qt.AlignCenter, "↘")

    def generate_and_save_mask(self):
        """
        根据 selection_rects 生成遮罩图片并保存到 mask_path。
        遮罩图片大小与视频帧一致，选区为255，其余为0。
        """
        if self.current_frame is None or self.frame_width <= 1 or self.frame_height <= 1:
            print("当前帧无效，无法生成遮罩")
            return

        mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        for rect in self.selection_rects:
            # 保证坐标在内容区域内
            left = max(0, rect.left())
            top = max(0, rect.top())
            right = min(self.frame_width, rect.right() + 1)
            bottom = min(self.frame_height, rect.bottom() + 1)
            mask[top:bottom, left:right] = 255

        # 保存遮罩图片
        return self.save_mask(mask)

    def save_mask(self, mask):
        # 获取视频文件路径和名称
        video_path = self.video_path  # 从父窗口获取视频路径
        video_dir = os.path.dirname(video_path)
        video_name = os.path.basename(video_path).split('.')[0]
        mask_path = os.path.join(video_dir, f"{video_name}_mask.png")
        
        # 确保路径使用正斜杠
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        if not os.access(os.path.dirname(mask_path), os.W_OK):
            print(f"没有权限写入路径: {os.path.dirname(mask_path)}")
            return None
        if mask is None or mask.size == 0:
            print("蒙版图为空")
            return None
        if mask.dtype != np.uint8:
            print(f"蒙版图数据类型错误: {mask.dtype}")
            return None 
        if not mask_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"文件名没有有效的扩展名: {mask_path}")
            return None 
        success = save_image_with_chinese_path(mask, mask_path)
        if success:
            print(f"蒙版图已成功保存到: {mask_path}")
        else:
            print(f"保存蒙版图失败: {mask_path}")
            return None 

        return mask_path

    