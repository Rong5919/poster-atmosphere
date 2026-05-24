# -*- coding: utf-8 -*-
"""
海报自动标注脚本
- 氛围标签：从文件夹名直接获取（最准确）
- 5个评分：基于图像特征自动计算（颜色丰富度、构图复杂度、风格特征、情感强度、商业属性）
"""

import os
import sys
import csv
import math
from PIL import Image
import numpy as np
from datetime import datetime

# 强制 UTF-8 输出
sys.stdout.reconfigure(encoding='utf-8')

# 配置
INPUT_DIR = r'C:\Users\admin\Desktop\RepViT\dataset\posters_resize'
OUTPUT_CSV = r'C:\Users\admin\Desktop\RepViT\dataset\label_auto.csv'

# 氛围标签映射（文件夹名 -> 标准标签）
FOLDER_TO_LABEL = {
    '卡通童趣海报': '卡通童趣',
    '喜庆热闹促销海报': '喜庆感',
    '复古国风海报': '复古国风',
    '时尚潮流商业海报': '时尚感',
    '科技感商业海报': '科技感',
    '简约商务品牌海报': '简约商务',
    '高端轻奢海报': '高端轻奢'
}

def analyze_color_richness(img_array):
    """分析颜色丰富度 (1-5分)"""
    # 计算颜色直方图
    hist_r = np.histogram(img_array[:,:,0], bins=256, range=(0,256))[0]
    hist_g = np.histogram(img_array[:,:,1], bins=256, range=(0,256))[0]
    hist_b = np.histogram(img_array[:,:,2], bins=256, range=(0,256))[0]
    
    # 计算有效颜色数（出现次数 > 0.1% 的颜色）
    threshold = img_array.shape[0] * img_array.shape[1] * 0.001
    unique_colors = (hist_r > threshold).sum() + (hist_g > threshold).sum() + (hist_b > threshold).sum()
    
    # 归一化到 1-5
    score = min(5, max(1, int(unique_colors / 100) + 1))
    return score

def analyze_composition_complexity(img_array):
    """分析构图复杂度 (1-5分)"""
    # 转灰度
    gray = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
    gray = gray.astype(np.uint8)
    
    # 边缘检测 (Sobel近似)
    gx = np.abs(np.diff(gray, axis=1)).sum()
    gy = np.abs(np.diff(gray, axis=0)).sum()
    edge_intensity = (gx + gy) / (gray.shape[0] * gray.shape[1])
    
    # 归一化到 1-5
    score = min(5, max(1, int(edge_intensity / 5) + 1))
    return score

def analyze_style_feature(img_array, label):
    """分析风格特征 (1-5分)"""
    h, w, _ = img_array.shape
    
    # 计算饱和度和亮度
    from colorsys import rgb_to_hsv
    pixels = img_array.reshape(-1, 3)
    
    saturations = []
    values = []
    for r, g, b in pixels[::100]:  # 采样加速
        h, s, v = rgb_to_hsv(r/255, g/255, b/255)
        saturations.append(s)
        values.append(v)
    
    avg_sat = np.mean(saturations)
    avg_val = np.mean(values)
    
    # 根据氛围标签调整评分
    if label in ['简约商务', '高端轻奢']:
        # 低饱和度、高亮度 = 高风格分
        score = min(5, max(1, int((1 - avg_sat) * 3 + avg_val * 2) + 1))
    elif label in ['卡通童趣', '喜庆感']:
        # 高饱和度 = 高风格分
        score = min(5, max(1, int(avg_sat * 5) + 1))
    else:
        # 中等标准
        score = min(5, max(1, int((avg_sat + avg_val) * 2.5) + 1))
    
    return score

def analyze_emotion_intensity(img_array, label):
    """分析情感强度 (1-5分)"""
    # 计算颜色对比度
    std_r = np.std(img_array[:,:,0])
    std_g = np.std(img_array[:,:,1])
    std_b = np.std(img_array[:,:,2])
    contrast = (std_r + std_g + std_b) / 3
    
    # 根据氛围标签调整
    if label in ['喜庆感', '卡通童趣']:
        score = min(5, max(1, int(contrast / 20) + 2))
    elif label in ['简约商务', '高端轻奢']:
        score = min(5, max(1, int(contrast / 25) + 1))
    else:
        score = min(5, max(1, int(contrast / 22) + 1))
    
    return score

def analyze_business_attribute(img_array, label):
    """分析商业属性 (1-5分)"""
    # 商业海报通常有：清晰文字区域（高对比度边缘）、专业配色
    
    # 计算整体亮度
    brightness = np.mean(img_array)
    
    # 计算色彩协调度（主色调占比）
    hist_r = np.histogram(img_array[:,:,0], bins=8, range=(0,256))[0]
    hist_g = np.histogram(img_array[:,:,1], bins=8, range=(0,256))[0]
    hist_b = np.histogram(img_array[:,:,2], bins=8, range=(0,256))[0]
    
    color_dominance = max(hist_r.max(), hist_g.max(), hist_b.max()) / (img_array.shape[0] * img_array.shape[1])
    
    # 根据氛围标签调整
    base_score = {
        '简约商务': 5,
        '科技感': 5,
        '高端轻奢': 5,
        '时尚感': 4,
        '喜庆感': 4,
        '复古国风': 3,
        '卡通童趣': 3
    }.get(label, 3)
    
    # 根据色彩协调度微调
    if color_dominance > 0.4:  # 主色调明显，更专业
        score = min(5, base_score)
    else:
        score = max(1, base_score - 1)
    
    return score

def process_image(img_path, label):
    """处理单张图片，返回5个评分"""
    try:
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)
        
        color = analyze_color_richness(img_array)
        composition = analyze_composition_complexity(img_array)
        style = analyze_style_feature(img_array, label)
        emotion = analyze_emotion_intensity(img_array, label)
        business = analyze_business_attribute(img_array, label)
        
        return color, composition, style, emotion, business
    except Exception as e:
        print(f'  [错误] {os.path.basename(img_path)}: {e}')
        return None

def main():
    print(f'\n{"="*60}')
    print(f'海报自动标注脚本 - 开始时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"="*60}\n')
    
    # 统计
    total_images = 0
    processed = 0
    errors = 0
    
    # 打开CSV文件
    with open(OUTPUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'atmosphere', 'color', 'composition', 'style', 'emotion', 'business'])
        
        # 遍历每个文件夹
        for folder_name in sorted(os.listdir(INPUT_DIR)):
            folder_path = os.path.join(INPUT_DIR, folder_name)
            
            if not os.path.isdir(folder_path):
                continue
            
            label = FOLDER_TO_LABEL.get(folder_name)
            if not label:
                print(f'[跳过] 未知文件夹: {folder_name}')
                continue
            
            images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            total_images += len(images)
            
            print(f'\n处理 [{label}] - {len(images)} 张图片')
            print('-' * 40)
            
            for i, img_name in enumerate(images, 1):
                img_path = os.path.join(folder_path, img_name)
                
                scores = process_image(img_path, label)
                if scores:
                    writer.writerow([img_name, label] + list(scores))
                    processed += 1
                else:
                    errors += 1
                
                # 进度显示
                if i % 50 == 0 or i == len(images):
                    print(f'  进度: {i}/{len(images)} ({i*100//len(images)}%)')
    
    # 输出统计
    print(f'\n{"="*60}')
    print(f'标注完成！')
    print(f'  - 总图片数: {total_images}')
    print(f'  - 成功处理: {processed}')
    print(f'  - 错误: {errors}')
    print(f'  - 输出文件: {OUTPUT_CSV}')
    print(f'  - 完成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"="*60}\n')

if __name__ == '__main__':
    main()
