
import sys
from typing import Union
import torch
import numpy as np
from PIL import Image
import ffmpeg

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from inpaint.sttn.sttn_inpaint import STTNVideoInpaint as STTN_PROCESSOR 
from inpaint.lama.lama_inpaint import LamaInpaint as LAMA_PROCESSOR
from inpaint.propainter.propainter import ProPainter as PROPAINTER_PROCESSOR



class InpaintManager:
    def __init__(self, video_path, save_folder, mask_path, mode=config.InpaintMode.STTN, callback=None) -> None:
        self.video_path = video_path
        self.save_folder = save_folder
        self.mask_path = mask_path
        self.callback = callback
        self.mode = mode

        # 设置输出视频文件的路径
        if save_folder:  # 需求2：如果save_folder不为空，则优先使用它
            os.makedirs(save_folder, exist_ok=True)  # 自动创建路径
            self.video_out_path = os.path.join(
                save_folder,
                f"{os.path.basename(self.video_path).rsplit('.', 1)[0]}_no_sub.mp4"
            )
        else:  
            self.video_out_path = os.path.join(
                os.path.dirname(os.path.abspath(self.video_path)),
                f"{os.path.basename(self.video_path).rsplit('.', 1)[0]}_no_sub.mp4"
            )
        

    def __call__(self):
        print("当前使用模型：",self.mode)

        processor = None
        if self.mode.upper() == config.InpaintMode.STTN.value.upper():
            processor = STTN_PROCESSOR(self.video_path,self.video_out_path,self.mask_path,self.callback)
        elif self.mode.upper() == config.InpaintMode.LAMA.value.upper() :
            processor = LAMA_PROCESSOR(self.video_path,self.video_out_path,self.mask_path,self.callback)
        elif self.mode.upper() == config.InpaintMode.PROPAINTER.value.upper():
            processor = PROPAINTER_PROCESSOR(self.video_path,self.video_out_path,self.mask_path,self.callback)
        else :
            processor = STTN_PROCESSOR(self.video_path,self.video_out_path,self.mask_path,self.callback)


        processor()
        self.replace_audio_of_b(self.video_path,self.video_out_path)
        self.callback(100)

    
    def replace_audio_of_b(self, video_a_path, video_b_path):
        """
        将视频A的音频复制到视频B，并覆盖保存视频B
        
        参数:
            video_a_path: 提供音频的视频路径
            video_b_path: 要被替换音频的视频路径（将直接被修改）
        """
        # 创建临时文件名
        temp_dir = os.path.dirname(video_b_path) or "."
        temp_path = os.path.join(temp_dir, f"temp_{os.path.basename(video_b_path)}")
        
        try:
            # 使用ffmpeg-python构建处理流程
            input_video = ffmpeg.input(video_b_path)
            input_audio = ffmpeg.input(video_a_path)
            
            (
                ffmpeg
                .input(video_b_path)['v']  # 仅视频流
                .output(
                    ffmpeg.input(video_a_path)['a'],  # 仅音频流
                    temp_path,
                    vcodec='copy',  # 复制视频
                    acodec='aac',   # 重新编码音频
                    shortest=None,  # 对齐时长
                    y=None          # 覆盖输出
                )
                .global_args('-loglevel', 'quiet')  # 完全静默
                .run(cmd='ffmpeg', capture_stdout=True, capture_stderr=True)
            )
            
            # 验证输出文件
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                raise RuntimeError("输出文件创建失败")
                
            # 替换原文件（原子操作）
            if os.name == 'nt':  # Windows系统
                os.remove(video_b_path)
            os.rename(temp_path, video_b_path)
            
        except ffmpeg.Error as e:
            print(f"FFmpeg处理失败: {e.stderr.decode('utf8') if e.stderr else e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            print(f"操作失败: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise


        

