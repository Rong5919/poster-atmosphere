# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\admin\Desktop\RepViT\dataset\label.csv', 'r', encoding='utf-8-sig') as f:
    lines = f.readlines()

print(f'Total lines: {len(lines)}')
for l in lines[:10]:
    print(l.strip())
