

from torch import cuda ,device
import os
from enum import Enum, unique




# 视频输出路径, 如果不设置，那默认就是原视频的路径
#VIDEO_OUT_FOLDER = r"F:\AI\lama\test_images"
VIDEO_OUT_FOLDER = None

device = device("cuda:0" if cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STTN_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sttn', 'infer_model.pth')
LAMA_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'big-lama', 'big-lama.pt')

# 这3个模型都是propainter用到
RAFT_MODEL_PATH = os.path.join(BASE_DIR, 'models',"propainter", 'raft-things.pth')
FLOW_MODEL_PATH = os.path.join(BASE_DIR, 'models',"propainter",  'recurrent_flow_completion.pth')
PROPAINTER_MODEL_PATH = os.path.join(BASE_DIR, 'models',"propainter",  'ProPainter.pth')

"""
4. STTN_MAX_LOAD_NUM
含义：STTN算法每次最多加载的视频帧数量
效果：设置越大速度越慢，但效果越好
注意：要保证STTN_MAX_LOAD_NUM大于STTN_NEIGHBOR_STRIDE和STTN_REFERENCE_LENGTH
"""
STTN_SKIP_DETECTION = True
# 参考帧步长
STTN_NEIGHBOR_STRIDE = 5
# 参考帧长度（数量）
STTN_REFERENCE_LENGTH = 10
# 设置STTN算法最大同时处理的帧数量
STTN_MAX_LOAD_NUM = 50
if STTN_MAX_LOAD_NUM < STTN_REFERENCE_LENGTH * STTN_NEIGHBOR_STRIDE:
    STTN_MAX_LOAD_NUM = STTN_REFERENCE_LENGTH * STTN_NEIGHBOR_STRIDE
# ×××××××××× InpaintMode.STTN算法设置 end ××××××××××


 
@unique
class InpaintMode(Enum):
    """
    图像重绘算法枚举
    """
    STTN = 'sttn'
    LAMA = 'lama'
    PROPAINTER = 'propainter'

INPAINT_MODE = InpaintMode.STTN


