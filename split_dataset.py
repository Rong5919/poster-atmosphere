# -*- coding: utf-8 -*-
"""
数据集划分脚本 - 纯 Python 实现
按 70% / 20% / 10% 划分训练集、验证集、测试集
"""

import sys
import csv
import random
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

random.seed(42)

# 读取数据并按标签分组
label_data = defaultdict(list)

with open(r'C:\Users\admin\Desktop\RepViT\dataset\label_auto.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        label_data[row['atmosphere']].append(row)

total = sum(len(v) for v in label_data.values())
print(f'总数据: {total} 条')
print('标签分布:')
for label, rows in sorted(label_data.items()):
    print(f'  {label}: {len(rows)}')

# 按比例划分
train_rows = []
val_rows = []
test_rows = []

for label, rows in label_data.items():
    random.shuffle(rows)
    n = len(rows)
    train_end = int(n * 0.7)
    val_end = int(n * 0.9)

    train_rows.extend(rows[:train_end])
    val_rows.extend(rows[train_end:val_end])
    test_rows.extend(rows[val_end:])

# 打乱顺序
random.shuffle(train_rows)
random.shuffle(val_rows)
random.shuffle(test_rows)

# 保存函数
def save_csv(rows, filename):
    path = f'C:\\Users\\admin\\Desktop\\RepViT\\dataset\\{filename}'
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'  已保存: {filename}')

# 保存
print('\n' + '=' * 50)
print('划分完成！')
save_csv(train_rows, 'train.csv')
save_csv(val_rows, 'val.csv')
save_csv(test_rows, 'test.csv')

print(f'\n  训练集: {len(train_rows)} 条 ({len(train_rows)*100//total}%)')
print(f'  验证集: {len(val_rows)} 条 ({len(val_rows)*100//total}%)')
print(f'  测试集: {len(test_rows)} 条 ({len(test_rows)*100//total}%)')
print('=' * 50)
