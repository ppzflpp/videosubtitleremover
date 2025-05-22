
from PyQt5.QtWidgets import (QApplication)
import sys
from PyQt5.QtGui import QFont
from  UI.VideoProcessor import VideoProcessor

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 获取屏幕分辨率
    screen = app.primaryScreen()
    size = screen.size()
    screenWidth = size.width()
    screenHeight = size.height()
    
    print(f"屏幕分辨率：({screenWidth},{screenHeight})")
    
    # 设置全局字体
    font = QFont("Microsoft YaHei", 10) 
    app.setFont(font)
    font.setBold(True)
    
    window = VideoProcessor(screenWidth,screenHeight)
    window.show()
    sys.exit(app.exec_())