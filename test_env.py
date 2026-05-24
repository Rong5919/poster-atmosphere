# -*- coding: utf-8 -*-
"""
测试 PyTorch 环境和数据集
"""
import sys
import os

# 设置输出编码
sys.stdout.reconfigure(encoding='utf-8')

print('=' * 60)
print('  环境测试')
print('=' * 60)

# 测试 PyTorch
try:
    import torch
    print(f'[√] PyTorch 版本: {torch.__version__}')
    print(f'[√] CUDA 可用: {torch.cuda.is_available()}')
except ImportError as e:
    print(f'[×] PyTorch 未安装: {e}')
    sys.exit(1)

# 测试其他依赖
try:
    import torchvision
    print(f'[√] torchvision 版本: {torchvision.__version__}')
except ImportError:
    print('[×] torchvision 未安装')

try:
    import timm
    print(f'[√] timm 版本: {timm.__version__}')
except ImportError:
    print('[!] timm 未安装，正在安装...')
    os.system('pip install timm')

try:
    import pandas
    print(f'[√] pandas 已安装')
except ImportError:
    print('[!] pandas 未安装')

try:
    from PIL import Image
    print(f'[√] Pillow 已安装')
except ImportError:
    print('[!] Pillow 未安装')

# 测试数据集
print('\n' + '=' * 60)
print('  数据集测试')
print('=' * 60)

try:
    from dataset import PosterDataset
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    train_csv = r'C:\Users\admin\Desktop\RepViT\dataset\train.csv'
    ds = PosterDataset(train_csv, transform=transform)
    
    print(f'[√] 训练集大小: {len(ds)}')
    
    # 测试加载一个样本
    img, label, scores = ds[0]
    print(f'[√] 图像 shape: {img.shape}')
    print(f'[√] 氛围标签: {ds.idx_to_label[label]}')
    print(f'[√] 评分: {scores.tolist()}')
    
except Exception as e:
    print(f'[×] 数据集加载失败: {e}')
    import traceback
    traceback.print_exc()

print('\n' + '=' * 60)
print('  测试完成！可以开始训练')
print('=' * 60)
