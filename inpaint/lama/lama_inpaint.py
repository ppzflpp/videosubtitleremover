import sys
from typing import Union
import torch
import numpy as np
from PIL import Image
import cv2

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from inpaint.utils.lama_util import prepare_img_and_mask
import config


class LamaInpaint:
    def __init__(self, video_path,video_out_path, mask_path=None, callback=None) -> None:
        # 视频和掩码路径
        self.video_path = video_path
        self.mask_path = mask_path
        self.callback = callback
        # 设置输出视频文件的路径
        self.video_out_path = video_out_path
        self.device = config.device
        model_path = os.path.join(config.LAMA_MODEL_PATH, 'big-lama.pt')
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        self.model.to(self.device)
        print("LAMA:video_path ",self.video_path)
        print("LAMA:video_out_path ",self.video_out_path)
        print("LAMA:mask_path ",self.mask_path)


    def __call__(self):
        # 读取视频帧信息
        reader, frame_info = self.read_frame_info_from_video()
        orig_width = frame_info['W_ori']
        orig_height = frame_info['H_ori']
        frame_count = frame_info['len']

        # 创建视频写入对象，用于输出修复后的视频
        writer = cv2.VideoWriter(self.video_out_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_info['fps'], (frame_info['W_ori'], frame_info['H_ori']))

        
        maskImage = Image.open(self.mask_path).convert("L")  
        index = 0
        while True:
            ret, frame = reader.read()
            if not ret:
                break

            index += 1
            image, mask = prepare_img_and_mask(frame, maskImage, self.device)
            with torch.inference_mode():
                inpainted = self.model(image, mask)
                cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
                cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
                cur_res = cur_res[:orig_height, :orig_width]
                writer.write(cur_res)
                
            self.callback(int(100 * float(index) / float(frame_count)))


    def read_frame_info_from_video(self):
        # 使用opencv读取视频
        reader = cv2.VideoCapture(self.video_path)
        # 获取视频的宽度, 高度, 帧率和帧数信息并存储在frame_info字典中
        frame_info = {
            'W_ori': int(reader.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5),  # 视频的原始宽度
            'H_ori': int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5),  # 视频的原始高度
            'fps': reader.get(cv2.CAP_PROP_FPS),  # 视频的帧率
            'len': int(reader.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)  # 视频的总帧数
        }
        # 返回视频读取对象、帧信息和视频写入对象
        return reader, frame_info