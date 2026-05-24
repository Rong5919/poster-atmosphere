# -*- coding: utf-8 -*-
"""
简化版训练脚本 - 专为 CPU 优化
使用 MobileViT-S 轻量模型
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

# 导入自定义模块
from dataset import PosterDataset
from model import SimpleCNN


def main():
    # ==================== 配置 ====================
    DATA_DIR = r'C:\Users\admin\Desktop\RepViT\dataset'
    OUTPUT_DIR = r'C:\Users\admin\Desktop\RepViT\outputs'
    
    # 超参数 (CPU优化版)
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    EPOCHS = 10  # CPU 训练，10个epoch足够
    IMG_SIZE = 224
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ==================== 数据加载 ====================
    print('\n加载数据集...')
    
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_ds = PosterDataset(os.path.join(DATA_DIR, 'train.csv'), transform=train_transform)
    val_ds = PosterDataset(os.path.join(DATA_DIR, 'val.csv'), transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f'训练集: {len(train_ds)} 样本')
    print(f'验证集: {len(val_ds)} 样本')
    
    # ==================== 模型初始化 ====================
    print('\n初始化模型...')
    
    model = SimpleCNN()
    model = model.to(device)
    
    # 加载已有模型继续训练
    checkpoint_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        print(f'加载已有模型: {checkpoint_path}')
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # 损失函数
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # ==================== 训练循环 ====================
    print('\n' + '=' * 60)
    print(f'开始训练 - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 60)
    
    best_val_acc = 0
    
    for epoch in range(EPOCHS):
        print(f'\nEpoch [{epoch+1}/{EPOCHS}]')
        print('-' * 40)
        
        # 训练
        model.train()
        train_loss = 0
        cls_correct = 0
        total_samples = 0
        
        for batch_idx, (imgs, labels, scores) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            scores = scores.to(device)
            
            optimizer.zero_grad()
            pred_cls, pred_reg = model(imgs)
            
            loss_cls = ce_loss(pred_cls, labels)
            loss_reg = mse_loss(pred_reg, scores)
            loss = loss_cls + 0.4 * loss_reg
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = pred_cls.max(1)
            cls_correct += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)
            
            if (batch_idx + 1) % 50 == 0:
                print(f'  Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}')
        
        train_acc = cls_correct / total_samples * 100
        
        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0
        
        with torch.no_grad():
            for imgs, labels, scores in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                scores = scores.to(device)
                
                pred_cls, pred_reg = model(imgs)
                
                loss_cls = ce_loss(pred_cls, labels)
                loss_reg = mse_loss(pred_reg, scores)
                loss = loss_cls + 0.4 * loss_reg
                
                val_loss += loss.item()
                _, predicted = pred_cls.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_samples += labels.size(0)
        
        val_acc = val_correct / val_samples * 100
        
        print(f'  训练 - Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%')
        print(f'  验证 - Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
            print(f'  ✓ 保存最佳模型 (Acc: {val_acc:.2f}%)')
    
    # ==================== 训练完成 ====================
    print('\n' + '=' * 60)
    print(f'训练完成！最佳验证准确率: {best_val_acc:.2f}%')
    print(f'模型保存路径: {OUTPUT_DIR}')
    print('=' * 60)


if __name__ == '__main__':
    main()
