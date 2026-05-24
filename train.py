# -*- coding: utf-8 -*-
"""
RepViT 氛围评价模型 - 训练脚本
多任务学习：分类（7种氛围）+ 回归（5个评分）
"""

import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

# 导入自定义模块
from dataset import PosterDataset
from model import RepViTAtmosphere


def train_one_epoch(model, dataloader, optimizer, ce_loss, mse_loss, device, epoch):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    cls_correct = 0
    total_samples = 0
    
    for batch_idx, (imgs, labels, scores) in enumerate(dataloader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        scores = scores.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        pred_cls, pred_reg = model(imgs)
        
        # 计算损失
        loss_cls = ce_loss(pred_cls, labels)
        loss_reg = mse_loss(pred_reg, scores)
        loss = loss_cls + 0.4 * loss_reg  # 分类损失权重更高
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = pred_cls.max(1)
        cls_correct += predicted.eq(labels).sum().item()
        total_samples += labels.size(0)
        
        # 打印进度
        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(dataloader):
            print(f'  Batch [{batch_idx+1}/{len(dataloader)}] '
                  f'Loss: {loss.item():.4f} '
                  f'(cls: {loss_cls.item():.4f}, reg: {loss_reg.item():.4f})')
    
    avg_loss = total_loss / len(dataloader)
    accuracy = cls_correct / total_samples * 100
    return avg_loss, accuracy


def validate(model, dataloader, ce_loss, mse_loss, device):
    """验证"""
    model.eval()
    total_loss = 0
    cls_correct = 0
    total_samples = 0
    reg_mae = 0
    
    with torch.no_grad():
        for imgs, labels, scores in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            scores = scores.to(device)
            
            pred_cls, pred_reg = model(imgs)
            
            loss_cls = ce_loss(pred_cls, labels)
            loss_reg = mse_loss(pred_reg, scores)
            loss = loss_cls + 0.4 * loss_reg
            
            total_loss += loss.item()
            _, predicted = pred_cls.max(1)
            cls_correct += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)
            reg_mae += torch.abs(pred_reg - scores).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = cls_correct / total_samples * 100
    avg_mae = reg_mae / (total_samples * 5)  # 每个评分的平均MAE
    
    return avg_loss, accuracy, avg_mae


def main():
    # ==================== 配置 ====================
    DATA_DIR = r'C:\Users\admin\Desktop\RepViT\dataset'
    OUTPUT_DIR = r'C:\Users\admin\Desktop\RepViT\outputs'
    PRETRAINED_PATH = r'C:\Users\admin\Desktop\RepViT\weights\repvit_m_1.0.pth'
    
    # 超参数 (CPU优化版)
    BATCH_SIZE = 2  # CPU下减小batch size
    LEARNING_RATE = 1e-4
    EPOCHS = 30
    NUM_WORKERS = 0  # Windows下建议设为0
    IMG_SIZE = 224  # 减小图像尺寸以节省内存
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    
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
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    print(f'训练集: {len(train_ds)} 样本')
    print(f'验证集: {len(val_ds)} 样本')
    
    # ==================== 模型初始化 ====================
    print('\n初始化模型...')
    
    # 使用 MobileViT-S（轻量级，适合 CPU）
    model = RepViTAtmosphere(model_name='mobilevit_s', pretrained=True)
    model = model.to(device)
    
    # 损失函数
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # ==================== 训练循环 ====================
    print('\n' + '=' * 60)
    print(f'开始训练 - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 60)
    
    best_val_acc = 0
    
    for epoch in range(EPOCHS):
        print(f'\nEpoch [{epoch+1}/{EPOCHS}]')
        print('-' * 40)
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, ce_loss, mse_loss, device, epoch
        )
        
        # 验证
        val_loss, val_acc, val_mae = validate(
            model, val_loader, ce_loss, mse_loss, device
        )
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印结果
        print(f'  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'  验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, MAE: {val_mae:.4f}')
        print(f'  学习率: {current_lr:.6f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_mae': val_mae,
            }, os.path.join(OUTPUT_DIR, 'best_model.pth'))
            print(f'  ✓ 保存最佳模型 (Acc: {val_acc:.2f}%)')
        
        # 每10个epoch保存一次
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), 
                      os.path.join(OUTPUT_DIR, f'model_epoch_{epoch+1}.pth'))
    
    # ==================== 训练完成 ====================
    print('\n' + '=' * 60)
    print(f'训练完成！最佳验证准确率: {best_val_acc:.2f}%')
    print(f'模型保存路径: {OUTPUT_DIR}')
    print('=' * 60)


if __name__ == '__main__':
    main()
