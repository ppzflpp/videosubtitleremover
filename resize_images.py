import os
import cv2
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def resize_image(input_path, output_path, target_width=960, target_height=540, quality=95):
    """
    缩放单张图片并保存
    :param input_path: 输入图片路径
    :param output_path: 输出图片路径
    :param target_width: 目标宽度
    :param target_height: 目标高度
    :param quality: 保存质量（1-100）
    """
    try:
        # 读取图片（自动处理所有格式：jpg/png/webp等）
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"警告：无法读取图片 {input_path}，可能不是有效的图像文件")
            return False

        # 保持透明通道（适用于PNG等）
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 计算缩放比例（保持宽高比）
        h, w = img.shape[:2]
        scale = min(target_width / w, target_height / h)
        new_size = (int(w * scale), int(h * scale))

        # 高质量缩放（LANCZOS插值）
        resized = cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)

        # 创建目标目录（如果不存在）
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 根据扩展名选择保存参数
        ext = os.path.splitext(output_path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            cv2.imwrite(output_path, cv2.cvtColor(resized, cv2.COLOR_RGBA2BGR), 
                         [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        elif ext == '.png':
            cv2.imwrite(output_path, cv2.cvtColor(resized, cv2.COLOR_RGBA2BGRA), 
                         [int(cv2.IMWRITE_PNG_COMPRESSION), 9 - quality // 10])
        else:
            cv2.imwrite(output_path, cv2.cvtColor(resized, cv2.COLOR_RGBA2BGR))
        return True
    except Exception as e:
        print(f"处理 {input_path} 时出错: {str(e)}")
        return False

def batch_resize_images(input_dir, output_dir, target_size=(960, 540), workers=4):
    """
    批量缩放图片
    :param input_dir: 输入目录
    :param output_dir: 输出目录
    :param target_size: 目标尺寸 (width, height)
    :param workers: 线程数
    """
    # 支持的图片格式
    valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}

    # 收集所有图片文件
    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_exts:
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                image_files.append((input_path, output_path))

    if not image_files:
        print(f"未在 {input_dir} 中找到图片文件")
        return

    print(f"找到 {len(image_files)} 张图片，开始缩放...")

    # 多线程处理
    success_count = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for input_path, output_path in image_files:
            futures.append(executor.submit(
                resize_image, 
                input_path, 
                output_path, 
                target_size[0], 
                target_size[1]
            ))

        # 进度条显示
        for future in tqdm(futures, desc="处理进度", unit="img"):
            success_count += 1 if future.result() else 0

    print(f"处理完成！成功: {success_count}/{len(image_files)}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='批量缩放图片到540p')
    parser.add_argument('input_dir', help='输入目录路径')
    parser.add_argument('output_dir', help='输出目录路径')
    parser.add_argument('--width', type=int, default=360, help='目标宽度')
    parser.add_argument('--height', type=int, default=720, help='目标高度')
    parser.add_argument('--workers', type=int, default=4, help='线程数（默认4）')
    
    args = parser.parse_args()

    batch_resize_images(
        args.input_dir,
        args.output_dir,
        (args.width, args.height),
        args.workers
    )