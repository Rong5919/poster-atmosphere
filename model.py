# -*- coding: utf-8 -*-
"""
轻量级 CNN 模型 - 无需预训练权重
多任务输出：7分类（氛围）+ 5回归（评分）
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    简单 CNN 模型，无需预训练
    - 适合小数据集和 CPU 训练
    """
    
    def __init__(self, num_classes=7, num_scores=5):
        super().__init__()
        
        # 卷积特征提取层
        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4: 28 -> 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 5: 14 -> 7
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # 分类头：7种氛围
        self.cls_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # 回归头：5个评分
        self.reg_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_scores)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: 输入图像 [B, 3, 224, 224]
        Returns:
            cls_out: 分类输出 [B, 7]
            reg_out: 回归输出 [B, 5], 范围 [1, 5]
        """
        # 特征提取
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        
        # 多任务输出
        cls_out = self.cls_head(x)
        reg_out = self.reg_head(x)
        
        # 将回归输出限制在 [1, 5] 范围
        reg_out = torch.sigmoid(reg_out) * 4 + 1
        
        return cls_out, reg_out


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    print('测试 SimpleCNN 模型...')
    
    model = SimpleCNN()
    
    # 随机输入
    x = torch.randn(2, 3, 224, 224)
    
    # 前向传播
    with torch.no_grad():
        cls_out, reg_out = model(x)
    
    print(f'输入shape: {x.shape}')
    print(f'分类输出shape: {cls_out.shape}')
    print(f'回归输出shape: {reg_out.shape}')
    print(f'回归输出范围: [{reg_out.min():.2f}, {reg_out.max():.2f}]')
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f'参数量: {total_params:,} ({total_params/1e6:.2f}M)')
    print('模型测试成功！')
