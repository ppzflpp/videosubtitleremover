# -*- coding: utf-8 -*-

import os
from pathlib import Path
import random
from moviepy import VideoFileClip
from PIL import Image, ImageDraw, ImageFont,ImageFilter,ImageChops
import numpy as np
import cv2
from matplotlib import font_manager

from .common_characters import common_characters

def random_chinese_character():

    # 随机选择一个元素
    index = random.randint(0, len(common_characters) - 1)
    
    # 获取对应的汉字字符
    return chr(common_characters[index])

def generate_random_chinese_text(max_length=30):
    """随机生成指定最大长度的汉字文本"""
    length = random.randint(1, max_length)  # 随机生成 1 到 max_length 个汉字
    text = ''.join(random_chinese_character() for _ in range(length))
    
    return text

def random_color(alpha=True):
    if alpha:
        return (random.randint(0, 255), random.randint(0, 255), random.randint(128, 255))
    else:
        return (random.randint(0, 255), random.randint(0, 255), random.randint(255, 255))

def random_font_size():
    return random.randint(10, 30)  # 调整字体大小范围

def random_position(width, height):
    return (random.randint(0, width - 200), random.randint(0, height - 100))  # 调整位置范围以适应新的字体大小

def random_shadow_offset():
    return (random.randint(-5, 5), random.randint(-5, 5))

def get_default_font_path():
    # 使用 matplotlib 的 font_manager 找到系统默认字体路径
    #font_path = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')[0]
    font_path = "C:/Windows/Fonts/msyh.ttc"  # 黑体字体路径
    return font_path

def wrap_text(text, font, max_width):
    """将文本自动换行，确保不超过最大宽度"""
    lines = []
    words = list(text)
    if not words:  # 处理空文本情况
        return ""
    current_line = words[0]
    for word in words[1:]:
        test_line = current_line + word
        if font.getlength(test_line) <= max_width - 20:  # 使用getlength代替getsize
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    return "\n".join(lines)

def add_random_subtitle_to_frame(frame,index, width, height):
    # Create a blank image with the same size as the frame
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Random font size and color
    font_size = random_font_size()
    font_color = random_color(False)
    shadow_color = (0, 0, 0, 128)  # Semi-transparent black for shadow
    border_color = random_color(False)  # 随机边框颜色
    background_color = random_color(False)  # 随机背景颜色

    # Random position
    position = random_position(width, height)

    # Random shadow offset
    shadow_offset = random_shadow_offset()

    # Random text
    text = generate_random_chinese_text()

    # Load the default system font and set font size
    font_path = get_default_font_path()
    font = ImageFont.truetype(font_path, font_size)

    # Wrap text to fit within the image width
    wrapped_text = wrap_text(text, font, width - position[0])

    # Calculate text size for each line
    lines = wrapped_text.split('\n')
    text_width = max(font.getlength(line) for line in lines)  # 使用getlength
    line_height = font.getmetrics()[0]  # 获取行高
    text_height = len(lines) * (line_height + 2)  # 计算总高度，包括行间距

    # Ensure the text fits within the image
    if position[1] + text_height > height:
        position = (position[0], height - text_height - 10)  # Adjust vertical position

    # Draw background rectangle
    #draw.rectangle([position[0]-10, position[1]-10, position[0] + text_width + 10 , position[1] + text_height + 10], fill=background_color)

    # 创建文本区域的mask图（1通道，0-255）
    mask = np.zeros((height, width), dtype=np.uint8)

    # Draw shadow
    border_width = 2  # 控制描边粗细
    current_y = position[1]
    for line in lines:
        temp_img = Image.new("RGBA", img.size, (0,0,0,0))
        temp_draw = ImageDraw.Draw(temp_img)
        
        # 先绘制文字（作为蒙版）
        temp_draw.text((position[0], current_y), line, font=font, fill=(255,255,255,255))
        
        # 使用3x3最大滤波膨胀（安全尺寸）
        dilated = temp_img.filter(ImageFilter.MaxFilter(3))
        
        # 描边 = 膨胀部分 - 原始文字
        border_mask = ImageChops.difference(dilated, temp_img)
        img.paste(border_color, (0,0), border_mask)
        
        # 绘制原始文字
        draw.text((position[0], current_y), line, font=font, fill=font_color)

        # 计算每行文本的边界框
        bbox = draw.textbbox((position[0], current_y), line, font=font)
        # 扩展边界（可选，根据描边粗细调整）
        bbox = (
            bbox[0] - border_width, 
            bbox[1] - border_width,
            bbox[2] + border_width,
            bbox[3] + border_width
        )
        # 在mask上绘制白色矩形
        cv2.rectangle(
            mask, 
            (bbox[0], bbox[1]), 
            (bbox[2], bbox[3]), 
            255, 
            -1  # 填充整个矩形
        )

        current_y += line_height + 2

    # Draw border
    #draw.rectangle([position[0]-10, position[1]-10, position[0] + text_width + 10, position[1] + text_height + 10], outline=border_color, width=2)

    # Convert PIL image to numpy array
    subtitle_layer = np.array(img)

    # Ensure the subtitle layer has the same number of channels as the frame
    if frame.shape[2] == 3:  # If the frame is RGB
        # 如果原图是 RGB，且字幕层是 RGBA，提取 RGB 部分并按 alpha 混合
        if subtitle_layer.shape[2] == 4:
            alpha = subtitle_layer[:, :, 3:] / 255.0  # 归一化 alpha
            subtitle_rgb = subtitle_layer[:, :, :3]
            # 完全覆盖（如果 alpha=1）或混合（如果 alpha<1）
            frame = (frame * (1 - alpha) + subtitle_rgb * alpha).astype(np.uint8)
        else:
            # 如果字幕层是 RGB，直接覆盖
            frame[:subtitle_layer.shape[0], :subtitle_layer.shape[1]] = subtitle_layer
    elif frame.shape[2] == 4:  # If the frame is RGBA
        # 如果原图是 RGBA，且字幕层是 RGBA，按 alpha 混合
        if subtitle_layer.shape[2] == 4:
            alpha = subtitle_layer[:, :, 3:] / 255.0
            subtitle_rgb = subtitle_layer[:, :, :3]
            frame_rgb = frame[:, :, :3]
            frame[:, :, :3] = (frame_rgb * (1 - alpha) + subtitle_rgb * alpha).astype(np.uint8)
        else:
            # 如果字幕层是 RGB，直接覆盖 RGB 部分
            frame[:subtitle_layer.shape[0], :subtitle_layer.shape[1], :3] = subtitle_layer

    if False:
        output_path = os.path.join(create_sttn_mask_folder(),f"frame_{index:04d}.png")
        cv2.imwrite(output_path.replace(f".png", "_mask.png"), mask)  # 保存mask图
        cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert to BGR before saving

    return frame, mask


def create_sttn_mask_folder():
    # 获取桌面路径（跨平台）
    desktop_path = Path.home() / 'Desktop'  # 适用于大多数系统
    
    # 特殊处理Linux系统（可能使用不同桌面目录名）
    if os.name == 'posix' and not (desktop_path / 'Desktop').exists():
        # 尝试常见的Linux桌面目录名
        for dirname in ['Desktop', 'desktop', 'Escritorio']:
            possible_path = Path.home() / dirname
            if possible_path.exists():
                desktop_path = possible_path
                break
    
    # 创建目标文件夹路径
    target_folder = desktop_path / 'sttn_mask'
    
    # 创建文件夹（如果不存在）
    try:
        target_folder.mkdir(exist_ok=True)  # exist_ok=True 避免文件夹已存在时报错
        return str(target_folder)
    except Exception as e:
        print(f"创建文件夹失败: {e}")
        return None


if __name__ == "__main__":
    input_video = f"E:/DragonProject/AI/test_videos/en_path/5秒视频.mp4"  # Replace with your input video file
    output_folder = f"E:/DragonProject/AI/test_videos/en_path/frames"  # Replace with your output folder

    save_frames_as_images(input_video, output_folder)