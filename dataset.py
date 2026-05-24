# -*- coding: utf-8 -*-
"""
数据集加载类
用于加载海报数据集，返回图像、氛围标签和5个评分
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import csv

class PosterDataset(Dataset):
    def __init__(self, csv_path, img_dir=None, transform=None):
        """
        Args:
            csv_path: CSV文件路径 (train.csv/val.csv/test.csv)
            img_dir: 图片根目录，默认为 dataset/posters_resize
            transform: 图像变换
        """
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.img_dir = img_dir or r'C:\Users\admin\Desktop\RepViT\dataset\posters_resize'
        
        # 氛围标签映射
        self.label_map = {
            "科技感": 0,
            "时尚感": 1,
            "喜庆感": 2,
            "简约商务": 3,
            "复古国风": 4,
            "卡通童趣": 5,
            "高端轻奢": 6
        }
        
        # 反向映射（用于预测结果显示）
        self.idx_to_label = {v: k for k, v in self.label_map.items()}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 获取文件名
        name = self.df.iloc[idx, 0]
        
        # 根据氛围标签确定子文件夹
        atmosphere = self.df.iloc[idx, 1]
        folder_map = {
            "卡通童趣": "卡通童趣海报",
            "喜庆感": "喜庆热闹促销海报",
            "复古国风": "复古国风海报",
            "时尚感": "时尚潮流商业海报",
            "科技感": "科技感商业海报",
            "简约商务": "简约商务品牌海报",
            "高端轻奢": "高端轻奢海报"
        }
        
        folder_name = folder_map.get(atmosphere, atmosphere)
        img_path = os.path.join(self.img_dir, folder_name, name)
        
        # 加载图像
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        # 标签
        atmosphere_idx = self.label_map[atmosphere]
        
        # 5个评分 - 确保转换为数值类型
        scores_values = self.df.iloc[idx, 2:7].values.astype('float32')
        scores = torch.from_numpy(scores_values)
        
        return img, atmosphere_idx, scores


def get_dataloaders(data_dir, batch_size=16, num_workers=0):
    """
    获取训练、验证、测试数据加载器
    
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        num_workers: 数据加载线程数
    """
    from torchvision import transforms
    
    # 训练数据增强
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 验证/测试变换
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_ds = PosterDataset(os.path.join(data_dir, 'train.csv'), transform=train_transform)
    val_ds = PosterDataset(os.path.join(data_dir, 'val.csv'), transform=val_transform)
    test_ds = PosterDataset(os.path.join(data_dir, 'test.csv'), transform=val_transform)
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    # 测试数据集加载
    print('测试数据集加载...')
    ds = PosterDataset(r'C:\Users\admin\Desktop\RepViT\dataset\train.csv')
    print(f'训练集大小: {len(ds)}')
    
    # 测试获取一个样本
    img, label, scores = ds[0]
    print(f'图像shape: {img.size if hasattr(img, "size") else img.shape}')
    print(f'氛围标签: {ds.idx_to_label[label]}')
    print(f'5个评分: {scores}')
