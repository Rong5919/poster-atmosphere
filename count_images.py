# -*- coding: utf-8 -*-
import os
import sys

# 强制 UTF-8 输出
sys.stdout.reconfigure(encoding='utf-8')

root = r'C:\Users\admin\Desktop\RepViT\dataset\posters_resize'
dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

total = 0
for d in sorted(dirs):
    count = len(os.listdir(os.path.join(root, d)))
    total += count
    print(f'{d}: {count} 张')

print(f'\n总计: {total} 张图片, {len(dirs)} 个类别')
