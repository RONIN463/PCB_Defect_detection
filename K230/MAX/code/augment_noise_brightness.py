"""
PCB数据集 噪声+亮度增强 生成脚本
=================================
基于 images/ 目录下693张原始缺陷图片，生成噪声增强和亮度增强图片
- 噪声：高斯噪声(σ=0.01~0.05) + 椒盐噪声(p=0.02)，模拟工业摄像头噪声干扰
- 亮度：±20%随机变化，模拟产线光照变化
- 噪声和亮度分别独立生成，不叠加
- 输出目录结构与 rotation 一致
"""

import shutil
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance


def add_gaussian_noise(image, sigma):
    """添加高斯噪声，sigma范围0.01~0.05"""
    img_array = np.array(image).astype(np.float32)
    std = sigma * 255
    noise = np.random.normal(0, std, img_array.shape)
    noisy_img = img_array + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)


def add_salt_pepper_noise(image, prob=0.02):
    """添加椒盐噪声"""
    img_array = np.array(image).copy()
    h, w = img_array.shape[:2]

    num_salt = int(prob * h * w * 0.5)
    salt_coords = [np.random.randint(0, dim, num_salt) for dim in (h, w)]
    if img_array.ndim == 3:
        img_array[salt_coords[0], salt_coords[1], :] = 255
    else:
        img_array[salt_coords[0], salt_coords[1]] = 255

    num_pepper = int(prob * h * w * 0.5)
    pepper_coords = [np.random.randint(0, dim, num_pepper) for dim in (h, w)]
    if img_array.ndim == 3:
        img_array[pepper_coords[0], pepper_coords[1], :] = 0
    else:
        img_array[pepper_coords[0], pepper_coords[1]] = 0

    return Image.fromarray(img_array)


def adjust_brightness(image, factor):
    """调整亮度，±20%即factor在0.8~1.2之间"""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def main():
    base_dir = Path(r'd:\数字图像化处理\数据集\PCB_DATASET\PCB_DATASET\All')
    images_dir = base_dir / 'images'
    output_noise_dir = base_dir / 'noise'
    output_brightness_dir = base_dir / 'brightness'

    # 清理旧数据
    if output_noise_dir.exists():
        shutil.rmtree(output_noise_dir)
    if output_brightness_dir.exists():
        shutil.rmtree(output_brightness_dir)

    defect_types = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spurious_copper', 'Spur']

    # 创建目录结构（和rotation一样）
    for defect in defect_types:
        (output_noise_dir / f'{defect}_noise').mkdir(parents=True, exist_ok=True)
        (output_brightness_dir / f'{defect}_brightness').mkdir(parents=True, exist_ok=True)

    total = 0

    for defect_dir in images_dir.iterdir():
        if not defect_dir.is_dir():
            continue
        defect_type = defect_dir.name
        if defect_type not in defect_types:
            continue

        print(f"处理 {defect_type}...")
        img_files = sorted(list(defect_dir.glob('*.jpg')))

        for img_path in img_files:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            stem = img_path.stem

            # === 噪声增强：高斯噪声 + 椒盐噪声 ===
            sigma = random.uniform(0.01, 0.05)
            noisy_img = add_gaussian_noise(img, sigma)
            noisy_img = add_salt_pepper_noise(noisy_img, prob=0.02)
            noisy_img.save(output_noise_dir / f'{defect_type}_noise' / f'{stem}.jpg', quality=95)

            # === 亮度增强 ===
            factor = random.uniform(0.8, 1.2)
            bright_img = adjust_brightness(img, factor)
            bright_img.save(output_brightness_dir / f'{defect_type}_brightness' / f'{stem}.jpg', quality=95)

            total += 2

    print(f"\n完成！共生成 {total} 张增强图片")
    print(f"噪声图片: {output_noise_dir} ({total // 2}张)")
    print(f"亮度图片: {output_brightness_dir} ({total // 2}张)")


if __name__ == '__main__':
    main()
