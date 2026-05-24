# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

csv_path = r'C:\Users\admin\Desktop\RepViT\dataset\label_auto.csv'
try:
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    print(f'Total lines: {len(lines)}')
    print('\nFirst 15 lines:')
    for l in lines[:15]:
        print(l.strip())
    
    # 统计每个标签的数量
    from collections import Counter
    labels = [l.split(',')[1] for l in lines[1:] if ',' in l]
    print('\n标签统计:')
    for label, count in Counter(labels).items():
        print(f'  {label}: {count}')
except FileNotFoundError:
    print('文件不存在，标注可能未完成')
except Exception as e:
    print(f'错误: {e}')
