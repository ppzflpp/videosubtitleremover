
import sys
from typing import Union
import torch
import numpy as np
from PIL import Image

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from inpaint.sttn.sttn_inpaint import STTNVideoInpaint as STTN_PROCESSOR 
from inpaint.lama.lama_inpaint import LamaInpaint as LAMA_PROCESSOR
from inpaint.propainter.propainter import ProPainter as PROPAINTER_PROCESSOR


from moviepy import VideoFileClip, AudioFileClip


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

    
    def replace_audio_of_b(self,video_a_path, video_b_path):
        """
        将视频A的音频复制到视频B，并覆盖保存视频B
        
        参数:
            video_a_path: 提供音频的视频路径
            video_b_path: 要被替换音频的视频路径（将直接被修改）
        """
        # 创建临时文件名
        temp_path = os.path.splitext(video_b_path)[0] + "_temp.mp4"
        
        try:
            # 加载视频B的视频流和视频A的音频流
            video_clip = VideoFileClip(video_b_path)
            audio_clip = AudioFileClip(video_a_path)
            
            # 设置新音频
            final_clip = video_clip.with_audio(audio_clip)
            
            # 输出到临时文件
            final_clip.write_videofile(
                temp_path,
                codec="libx264",
                audio_codec="aac",
                threads=4,
                logger=None
            )
            
            # 关闭资源
            video_clip.close()
            audio_clip.close()
            final_clip.close()
            
            # 删除原文件并将临时文件重命名为原文件
            os.remove(video_b_path)
            os.rename(temp_path, video_b_path)          
            
        except Exception as e:
            # 如果出错，删除临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
            print(f"操作失败: {str(e)}")


        

