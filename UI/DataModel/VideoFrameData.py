class VideoFrameData():
    def __init__(self,screenWidth,screenHeight,frameWidth,frameHeight,fps,frameCount,videoPath,maskPath,savePath,algo_mode):
        self.screenWidth = screenWidth
        self.screenHeight = screenHeight
        self.frameWidth = frameHeight
        self.fps = fps
        self.videoPath = videoPath
        self.maskPath = maskPath
        self.savePath = savePath
        self.algo_mode = algo_mode
        self.selection_rects = []  # 用于存储多个矩形区域（保存原始比例的坐标）

