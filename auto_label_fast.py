# -*- coding: utf-8 -*-
"""
海报自动标注脚本 - 优化版
- 氛围标签：从文件夹名直接获取
- 5个评分：基于图像特征快速计算
- 优化：采样像素加速计算
"""

import os
import sys
import csv
import numpy as np
from PIL import Image
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

INPUT_DIR = r'C:\Users\admin\Desktop\RepViT\dataset\posters_resize'
OUTPUT_CSV = r'C:\Users\admin\Desktop\RepViT\dataset\label_auto.csv'

FOLDER_TO_LABEL = {
    '卡通童趣海报': '卡通童趣',
    '喜庆热闹促销海报': '喜庆感',
    '复古国风海报': '复古国风',
    '时尚潮流商业海报': '时尚感',
    '科技感商业海报': '科技感',
    '简约商务品牌海报': '简约商务',
    '高端轻奢海报': '高端轻奢'
}

def analyze_image_fast(img_array, label):
    """快速分析图像，返回5个评分"""
    h, w, c = img_array.shape
    
    # 采样加速（每10个像素取1个）
    sampled = img_array[::10, ::10]
    pixels = sampled.reshape(-1, 3)
    
    # 1. 颜色丰富度
    std_colors = [np.std(pixels[:, i]) for i in range(3)]
    color_score = min(5, max(1, int(np.mean(std_colors) / 25) + 1))
    
    # 2. 构图复杂度（边缘强度）
    gray = 0.299 * sampled[:,:,0] + 0.587 * sampled[:,:,1] + 0.114 * sampled[:,:,2]
    edge_v = np.abs(np.diff(gray, axis=0)).sum()
    edge_h = np.abs(np.diff(gray, axis=1)).sum()
    edge_score = min(5, max(1, int((edge_v + edge_h) / (gray.size * 2) * 10) + 1))
    
    # 3. 风格特征（基于饱和度）
    from colorsys import rgb_to_hsv
    saturations = []
    for r, g, b in pixels[::10]:
        _, s, _ = rgb_to_hsv(r/255, g/255, b/255)
        saturations.append(s)
    avg_sat = np.mean(saturations)
    
    if label in ['简约商务', '高端轻奢']:
        style_score = min(5, max(1, int((1 - avg_sat) * 4) + 2))
    elif label in ['卡通童趣', '喜庆感']:
        style_score = min(5, max(1, int(avg_sat * 5) + 1))
    else:
        style_score = min(5, max(1, int(avg_sat * 3) + 2))
    
    # 4. 情感强度（色彩对比）
    contrast = np.mean([np.std(pixels[:, i]) for i in range(3)])
    emotion_score = min(5, max(1, int(contrast / 30) + 1))
    
    # 5. 商业属性（基于标签）
    business_map = {
        '简约商务': 5, '科技感': 5, '高端轻奢': 5,
        '时尚感': 4, '喜庆感': 4,
        '复古国风': 3, '卡通童趣': 3
    }
    business_score = business_map.get(label, 3)
    
    return color_score, edge_score, style_score, emotion_score, business_score

def main():
    print(f'\n{"="*60}')
    print(f'海报自动标注 - 优化版')
    print(f'开始时间: {datetime.now().strftime("%H:%M:%S")}')
    print(f'{"="*60}\n')
    
    total = 0
    processed = 0
    
    with open(OUTPUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'atmosphere', 'color', 'composition', 'style', 'emotion', 'business'])
        
        for folder_name in sorted(os.listdir(INPUT_DIR)):
            folder_path = os.path.join(INPUT_DIR, folder_name)
            if not os.path.isdir(folder_path):
                continue
            
            label = FOLDER_TO_LABEL.get(folder_name)
            if not label:
                continue
            
            images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            total += len(images)
            
            print(f'[{label}] {len(images)} 张...', end=' ', flush=True)
            start = datetime.now()
            
            for img_name in images:
                try:
                    img_path = os.path.join(folder_path, img_name)
                    img = Image.open(img_path).convert('RGB')
                    img_array = np.array(img)
                    
                    scores = analyze_image_fast(img_array, label)
                    writer.writerow([img_name, label] + list(scores))
                    processed += 1
                except Exception as e:
                    pass
            
            elapsed = (datetime.now() - start).total_seconds()
            print(f'完成 ({elapsed:.1f}s)')
    
    print(f'\n{"="*60}')
    print(f'完成! 总计: {processed}/{total}')
    print(f'输出: {OUTPUT_CSV}')
    print(f'{"="*60}\n')

if __name__ == '__main__':
    main()
