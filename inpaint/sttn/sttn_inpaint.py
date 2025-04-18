import copy
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from typing import List
import sys
import shutil
from tqdm import tqdm

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from inpaint.sttn.auto_sttn import InpaintGenerator
from inpaint.utils.sttn_utils import Stack, ToTorchFormatTensor

from inpaint.utils.utils import read_image_with_chinese_path

# 定义图像预处理方式
_to_tensors = transforms.Compose([
    Stack(),  # 将图像堆叠为序列
    ToTorchFormatTensor()  # 将堆叠的图像转化为PyTorch张量
])


class STTNInpaint:

    def __init__(self,callback=None):
        # 模型输入用的宽和高
        self.model_input_width, self.model_input_height = 432, 240
        self.device = config.device

        # 1. 创建InpaintGenerator模型实例并装载到选择的设备上
        self.model = InpaintGenerator(self.model_input_width,self.model_input_height).to(self.device)

        # 2. 载入预训练模型的权重，转载模型的状态字典
        self.model.load_state_dict(torch.load(config.STTN_MODEL_PATH, map_location=self.device)['netG'])
        # 3. 将模型设置为评估模式
        self.model.eval()
        # 2. 设置相连帧数
        self.neighbor_stride = config.STTN_NEIGHBOR_STRIDE
        self.ref_length = config.STTN_REFERENCE_LENGTH
        self.callback = callback

    def get_cropped_area(self, mask, use_all=False):
        # 确保mask的形状是(H, W)
        if len(mask.shape) == 3 and mask.shape[2] == 1:
            mask = mask[:, :, 0]

        # 如果use_all为True，直接返回整个图像区域
        if use_all:
            return 0, mask.shape[0], 0, mask.shape[1]
        
        # 获取mask图的边界框
        y, x = np.where(mask > 0)
        if y.size == 0 or x.size == 0:
            # 如果mask中没有非零元素，返回整个图像区域
            return 0, mask.shape[0], 0, mask.shape[1]
        
        ymin, ymax = np.min(y), np.max(y)
        xmin, xmax = np.min(x), np.max(x)
        
        # 计算裁剪区域
        h, w = ymax - ymin, xmax - xmin
        scale_size_w = w * config.STTN_ORI_IMAGE_FAC
        scale_size_h = h * config.STTN_ORI_IMAGE_FAC
        ycenter, xcenter = (ymin + ymax) // 2, (xmin + xmax) // 2
        crop_ymin = max(0, ycenter - h // 2 - scale_size_h)
        crop_ymax = min(mask.shape[0], ycenter + h // 2 + scale_size_h)
        crop_xmin = max(0, xcenter - w // 2 - scale_size_w)
        crop_xmax = min(mask.shape[1], xcenter + w // 2 + scale_size_w)
        return crop_ymin, crop_ymax, crop_xmin, crop_xmax

    def __call__(self, input_frames: List[np.ndarray], input_mask: np.ndarray):
        """
        :param input_frames: 原视频帧
        :param input_mask: 字幕区域mask
        """
        _, mask = cv2.threshold(input_mask, 127, 1, cv2.THRESH_BINARY)
        mask = mask[:, :, None]
        H_ori, W_ori = mask.shape[:2]
        H_ori = int(H_ori + 0.5)
        W_ori = int(W_ori + 0.5)
        # 计算裁剪区域
        crop_ymin, crop_ymax, crop_xmin, crop_xmax = self.get_cropped_area(mask,True) 
        # 初始化帧存储变量
        frames_hr = copy.deepcopy(input_frames)
        cropped_frames = []  # 存放裁剪后帧的列表
        # 读取并裁剪帧
        for frame in frames_hr:
            cropped_frame = frame[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            cropped_frames.append(cropped_frame)
        # 调用inpaint函数进行处理
        inpainted_frames = self.inpaint(cropped_frames, mask)
        # 将修复后的帧放回原图
        for i in range(len(frames_hr)):
            frames_hr[i][crop_ymin:crop_ymax, crop_xmin:crop_xmax] = inpainted_frames[i]
        return frames_hr

    @staticmethod
    def read_mask(path):
        img = read_image_with_chinese_path(path)
        # 转为binary mask
        ret, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        img = img[:, :, None]
        return img

    def get_ref_index(self, neighbor_ids, length):
        """
        采样整个视频的参考帧
        """
        # 初始化参考帧的索引列表
        ref_index = []
        # 在视频长度范围内根据ref_length逐步迭代
        for i in range(0, length, self.ref_length):
            # 如果当前帧不在近邻帧中
            if i not in neighbor_ids:
                # 将它添加到参考帧列表
                ref_index.append(i)
        # 返回参考帧索引列表
        return ref_index

    def inpaint(self, frames: List[np.ndarray], mask: np.ndarray):
        start_time = time.time()

        """
        使用STTN完成空洞填充（空洞即被遮罩的区域）
        """
        if config.DEBUG:
            shutil.rmtree(config.DEBUG_DIR_CROP_FRAME_INPUT, ignore_errors=True)
            os.makedirs(config.DEBUG_DIR_CROP_FRAME_INPUT, exist_ok=True)
            for i, frame in enumerate(frames):
                if i % 10 == 0:  # 每10帧保存一次
                    save_path = os.path.join(config.DEBUG_DIR_CROP_FRAME_INPUT, f"crop_frame_input_{i:04d}.png")
                    cv2.imwrite(save_path, frame)

            # 假设mask是0和1的二值图，乘以255以便保存为图片
            cv2.imwrite(os.path.join(config.DEBUG_DIR_CROP_FRAME_INPUT,"crop_frame_intput_mask.png"), mask * 255)

        frame_length = len(frames)
        original_height, original_width = frames[0].shape[:2]  

        # 缩放 frames 和 mask 到模型需要的尺寸
        resized_frames = [None] * frame_length
        for i, frame in enumerate(frames):
            resized_frames[i] = cv2.resize(frame, (self.model_input_width, self.model_input_height), 
                                        interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(mask, (self.model_input_width, self.model_input_height), interpolation=cv2.INTER_NEAREST)

        # 预处理二进制掩码（提前计算）
        binary_mask = (resized_mask.squeeze() != 0).astype(np.uint8)
        binary_mask = np.stack([binary_mask]*3, axis=-1)  # [H,W,3]

        if config.DEBUG:
            shutil.rmtree(config.DEBUG_DIR_MODEL_INPUT, ignore_errors=True)
            os.makedirs(config.DEBUG_DIR_MODEL_INPUT, exist_ok=True)
            for i, frame in enumerate(resized_frames):
                if i % 10 == 0:
                    save_path = os.path.join(config.DEBUG_DIR_MODEL_INPUT, f"model_frame_input_{i:04d}.png")
                    cv2.imwrite(save_path, frame)

            # 假设mask是0和1的二值图，乘以255以便保存为图片
            cv2.imwrite(os.path.join(config.DEBUG_DIR_MODEL_INPUT,"model_frame_intput_mask.png"), resized_mask * 255)

        # 对帧进行预处理转换为张量，并进行归一化
        feats = _to_tensors(resized_frames).unsqueeze(0) * 2 - 1
        mask_tensor = _to_tensors([resized_mask]).unsqueeze(0)  

        # 扩展mask的批次维度
        mask_tensor = mask_tensor.expand(-1,feats.size(1), -1, -1, -1)  

        # 把特征张量转移到指定的设备（CPU或GPU）
        feats, mask_tensor = feats.to(self.device), mask_tensor.to(self.device)

        end_time = time.time()
        if config.DEBUG:
            print("STTNInpaint,【修复中】预处理1耗时：",(end_time - start_time))

        # 初始化一个与视频长度相同的列表，用于存储处理完成的帧
        comp_frames = [None] * frame_length
        
        # 关闭梯度计算，用于推理阶段节省内存并加速
        with torch.no_grad():
            # 将处理好的帧通过编码器，产生特征表示
            feats = self.model.encoder((feats * (1 - mask_tensor).float()).view(frame_length, 3, self.model_input_height, self.model_input_width))
            # 获取特征维度信息
            _, c, feat_h, feat_w = feats.size()
            # 调整特征形状以匹配模型的期望输入
            feats = feats.view(1, frame_length, c, feat_h, feat_w)

        end_time = time.time()
        if config.DEBUG:
            print("STTNInpaint,【修复中】预处理2耗时：",(end_time - start_time))

        # 获取重绘区域
        # 在设定的邻居帧步幅内循环处理视频
        for f in range(0, frame_length, self.neighbor_stride):
            end_time = time.time()

            # 计算邻近帧的ID
            neighbor_ids = [i for i in range(max(0, f - self.neighbor_stride), min(frame_length, f + self.neighbor_stride + 1))]
            # 获取参考帧的索引
            ref_ids = self.get_ref_index(neighbor_ids, frame_length)
            
            if config.DEBUG:
                print(f"STTNInpaint,【修复中】修复第{f}帧,【步骤1】耗时： :  {(time.time() - end_time):.2f}秒")
                end_time = time.time()

            # 同样关闭梯度计算
            with torch.no_grad():
                if config.DEBUG:
                    print(f"STTNInpaint,【修复中】修复第{f}帧,【步骤2】耗时： :  {(time.time() - end_time):.2f}秒")
                    end_time = time.time()

                # 通过模型推断特征并传递给解码器以生成完成的帧
                #pred_feat = self.model.infer(feats[0, neighbor_ids + ref_ids, :, :, :], mask_tensor[0, neighbor_ids + ref_ids, :, :, :])
                # 使用内存高效的推断方式
                pred_feat = self.model.infer(
                    feats[0, neighbor_ids + ref_ids], 
                    mask_tensor[0, neighbor_ids + ref_ids]
                )

                if config.DEBUG:
                    print(f"STTNInpaint,【修复中】修复第{f}帧,【步骤3耗】时： :  {(time.time() - end_time):.2f}秒")
                    end_time = time.time()

                # 将预测的特征通过解码器生成图片，并应用激活函数tanh，然后分离出张量；解码并立即移回CPU
                pred_img = torch.tanh(self.model.decoder(pred_feat[:len(neighbor_ids)])).detach()
                pred_img = (pred_img.cpu() + 1) / 2 * 255
                pred_img = pred_img.permute(0, 2, 3, 1).numpy().astype(np.uint8)

                # 释放中间变量
                del pred_feat

                if config.DEBUG:
                    print(f"STTNInpaint,【修复中】修复第{f}帧,【步骤4】耗时： :  {(time.time() - end_time):.2f}秒")
                    end_time = time.time()

                # 混合处理
                for i, idx in enumerate(neighbor_ids):
                    img = np.array(pred_img[i]).astype(np.uint8) * binary_mask + resized_frames[idx] * (1 - binary_mask)
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img.astype(np.float32)
                    else:
                        cv2.addWeighted(comp_frames[idx], 0.5, img.astype(np.float32), 0.5, 0, 
                                    dst=comp_frames[idx])

        if config.DEBUG:
            # 保存处理后文件 调试用
            shutil.rmtree(config.DEBUG_DIR_MODEL_OUTPUT, ignore_errors=True)
            os.makedirs(config.DEBUG_DIR_MODEL_OUTPUT, exist_ok=True)
            for i, frame in enumerate(comp_frames):
                save_path = os.path.join(config.DEBUG_DIR_MODEL_OUTPUT, f"model_frame_output_{i:04d}.png")
                cv2.imwrite(save_path, frame)

        # 最终输出处理
        resized_comp_frames = [None] * frame_length
        for i, frame in enumerate(comp_frames):
            resized_frame = cv2.resize(frame.astype(np.uint8), (original_width, original_height), 
                                    interpolation=cv2.INTER_LINEAR)
            resized_comp_frames[i] = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)
        
        if config.DEBUG:
            # 保存处理后文件 调试用
            shutil.rmtree(config.DEBUG_DIR_CROP_FRAME_OUTPUT, ignore_errors=True)
            os.makedirs(config.DEBUG_DIR_CROP_FRAME_OUTPUT, exist_ok=True)
            for i, frame in enumerate(resized_comp_frames):
                save_path = os.path.join(config.DEBUG_DIR_CROP_FRAME_OUTPUT, f"crop_frame_output_{i:04d}.png")
                cv2.imwrite(save_path, frame)


        #释放内存，显存  
        torch.cuda.empty_cache()   
        del resized_frames, resized_mask  
        del feats, mask  

        return resized_comp_frames

class STTNVideoInpaint:

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

    def __init__(self, video_path, video_out_path, mask_path=None, callback=None):
        start_time = time.time()

        # STTNInpaint视频修复实例初始化
        self.sttn_inpaint = STTNInpaint(callback)
        # 视频和掩码路径
        self.video_path = video_path
        self.mask_path = mask_path
        self.callback = callback
        # 设置输出视频文件的路径
        self.video_out_path = video_out_path

        self.clip_gap = config.STTN_MAX_LOAD_NUM

        end_time = time.time()
        if config.DEBUG:
            print("STTNVideoInpaint,init耗时：",(end_time - start_time))

    def __call__(self, input_mask=None, input_sub_remover=None, tbar=None):
        start_time = time.time()
        # 读取视频帧信息
        reader, frame_info = self.read_frame_info_from_video()
        if input_sub_remover is not None:
            writer = input_sub_remover.video_writer
        else:
            # 创建视频写入对象，用于输出修复后的视频
            writer = cv2.VideoWriter(self.video_out_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_info['fps'], (frame_info['W_ori'], frame_info['H_ori']))
        # 读取掩码
        if input_mask is None:
            mask = self.sttn_inpaint.read_mask(self.mask_path)
        else:
            _, mask = cv2.threshold(input_mask, 127, 1, cv2.THRESH_BINARY)
            mask = mask[:, :, None]
        
        # 强制转为2维(H,W)
        mask = mask.squeeze()
        if len(mask.shape) == 3:  # 如果还是3维，取第一个通道
            mask = mask[:, :, 0]


        # 计算裁剪区域
        crop_ymin, crop_ymax, crop_xmin, crop_xmax = self.sttn_inpaint.get_cropped_area(mask,config.STTN_USE_ORI_IMAGE_FULL_SIZE)  
        # 裁剪mask图
        cropped_mask = mask[crop_ymin:crop_ymax, crop_xmin:crop_xmax]


        
        # 读取和修复高分辨率帧
        frames_hr = []  # 高分辨率帧列表
        cropped_frames = []  # 裁剪后的帧列表
        while True:
            success, frame = reader.read()
            if not success:
                break
            frames_hr.append(frame)
            # 裁剪原图
            cropped_frame = frame[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            cropped_frames.append(cropped_frame)

        end_time = time.time()
        if config.DEBUG:
            print("STTNVideoInpaint,修复准备耗时：",(end_time - start_time))
        
        # 新增分批次处理逻辑 ##############################################
        batch_size = config.STTN_MAX_LOAD_NUM
        inpainted_frames = []
        
        # 按50帧分批次处理
        count = len(cropped_frames) // batch_size + 1
        index = 0
        for batch_idx in tqdm(range(0, len(cropped_frames), batch_size), 
                    desc=f"处理中:共{len(cropped_frames)}帧"):
            batch_frames = cropped_frames[batch_idx:batch_idx + batch_size]
            # 调用inpaint方法处理当前批次
            batch_inpainted = self.sttn_inpaint.inpaint(batch_frames, cropped_mask)
            
            # 合并修复结果
            inpainted_frames.extend(batch_inpainted)

            #更新进度条
            index = index + 1
            if self.callback:
                progress = int( float(index) / float(count) * 100) 
                if progress > 90:
                    self.callback(90)
                else :
                    self.callback(progress)
        
        end_time = time.time()
        if config.DEBUG:
            print("STTNVideoInpaint,修复耗时：",(end_time - start_time))

        # 6. 精确合并修复区域
        for i in range(len(frames_hr)):
            # 获取当前帧和修复区域
            original_frame = frames_hr[i]
            repaired_region = inpainted_frames[i]
            
            # 创建边缘过渡区域（模糊mask边缘）
            blur_size = 5  # 可以调整这个值控制融合宽度
            blurred_mask = cv2.GaussianBlur(cropped_mask.astype(np.float32), (blur_size, blur_size), 0)
            
            # 对每个颜色通道处理
            for c in range(3):
                # 原始图像区域
                original_region = original_frame[crop_ymin:crop_ymax, crop_xmin:crop_xmax, c]
                
                # 加权融合
                blended_region = (original_region * (1 - blurred_mask) + 
                                repaired_region[:, :, c] * blurred_mask)
                
                # 只替换mask区域（>0的部分）
                original_frame[crop_ymin:crop_ymax, crop_xmin:crop_xmax, c] = np.where(
                    cropped_mask > 0,
                    blended_region,
                    original_region
                )
            
            writer.write(original_frame)

        # 释放视频写入对象
        writer.release()
        self.callback(95)

        end_time = time.time()
        if config.DEBUG:
            print("STTNVideoInpaint,修复后耗时：",(end_time - start_time))