

from torch import cuda ,device
import os
from enum import Enum, unique


device = device("cuda:0" if cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STTN_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sttn', 'infer_model.pth')
###这个是自己训练的模型
#STTN_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sttn', 'gen_00001.pth')
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

###训练sttn用到  一定要是 train/sttn/configs/youtube-vos.jos 里面w和h是  1/4关系； 这个值在train/sttn/model/sttn.py 里面用到
#patch_size = [(90, 180)]
###训练的时候需要注意的就是 configs文件夹下面的json配置文件 和  patch_size 是 1/4关系，可以训练高分辨率，但是容易OOM问题
###json 文件中 {"A":85,....} 代表数据集里面 A这个视频的所有视频帧（对应文件就是A.zip）,85代表这个视频一共85帧
###数据集需要放到 train/sttn/datasets/xxx 下面；  里面必须是所有图片的zip打包文件 , 序列帧命名方式  00000.jpg,00001.jpg ......
### 代码训练的逻辑就是生成某个视频所有帧的mask文件，mask文件时程序自动生成，随机挖洞，进行训练
###训练完成后，在train/sttn/release_model 文件下面生成训练后模型，其中gen_XXXXX.pth  模型可以用来进行推理


 
@unique
class InpaintMode(Enum):
    """
    图像重绘算法枚举
    """
    STTN = 'sttn'
    LAMA = 'lama'
    PROPAINTER = 'propainter'

INPAINT_MODE = InpaintMode.STTN


