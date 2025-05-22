

from PyQt5.QtCore import pyqtSignal,QThread
import time

from inpaint.InpaintManager import InpaintManager

class VideoProcessingThread(QThread):
    """处理视频的线程"""
    finished = pyqtSignal(str, int,int)  # 定义信号，用于通知主线程处理完成

    def __init__(self,parent, video_path, save_folder,mask_path,mode,child_mode):
        super().__init__()
        self.video_processor = parent
        self.video_path = video_path
        self.save_folder = save_folder
        self.mask_path = mask_path
        self.mode = mode
        self.child_mode = child_mode
    
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
        # 模拟耗时操作
        manager = InpaintManager(self.video_path, self.save_folder,self.mask_path,self.mode,self.child_mode,callback=self.update)
        manager()
        
        # 记录结束时间
        time_cost = int(time.time() - start)
        
        # 发射信号，通知主线程处理完成
        self.finished.emit(manager.video_out_path,time_cost,100)