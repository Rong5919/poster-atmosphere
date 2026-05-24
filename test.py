# -*- coding: utf-8 -*-
"""
RepViT 氛围评价模型 - 测试脚本
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import PosterDataset
from model import SimpleCNN

sys.stdout.reconfigure(encoding='utf-8')


def test_model(model_path, test_csv, device):
    """测试模型"""
    
    # 标签映射
    idx_to_label = {
        0: "科技感", 1: "时尚感", 2: "喜庆感", 3: "简约商务",
        4: "复古国风", 5: "卡通童趣", 6: "高端轻奢"
    }
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载测试集
    test_ds = PosterDataset(test_csv, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # 加载模型
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f'\n测试集大小: {len(test_ds)}')
    print('=' * 60)
    
    # 测试
    correct = 0
    total_mae = 0
    
    with torch.no_grad():
        for idx, (img, label, scores) in enumerate(test_loader):
            img = img.to(device)
            
            pred_cls, pred_reg = model(img)
            
            pred_label = pred_cls.argmax(dim=1).item()
            true_label = label.item()
            
            is_correct = pred_label == true_label
            correct += is_correct
            
            mae = torch.abs(pred_reg.cpu() - scores).mean().item()
            total_mae += mae
            
            # 显示前10个样本
            if idx < 10:
                print(f'\n样本 {idx+1}:')
                print(f'  真实氛围: {idx_to_label[true_label]}')
                print(f'  预测氛围: {idx_to_label[pred_label]} {"✓" if is_correct else "✗"}')
                print(f'  真实评分: {scores[0].tolist()}')
                print(f'  预测评分: {[f"{x:.1f}" for x in pred_reg[0].cpu().tolist()]}')
    
    # 统计
    accuracy = correct / len(test_ds) * 100
    avg_mae = total_mae / len(test_ds)
    
    print('\n' + '=' * 60)
    print('测试结果汇总:')
    print(f'  氛围分类准确率: {accuracy:.2f}% ({correct}/{len(test_ds)})')
    print(f'  评分平均MAE: {avg_mae:.4f}')
    print('=' * 60)


if __name__ == '__main__':
    MODEL_PATH = r'C:\Users\admin\Desktop\RepViT\outputs\best_model.pth'
    TEST_CSV = r'C:\Users\admin\Desktop\RepViT\dataset\test.csv'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    if os.path.exists(MODEL_PATH):
        test_model(MODEL_PATH, TEST_CSV, device)
    else:
        print(f'\n模型文件不存在: {MODEL_PATH}')
        print('请先运行 train_simple.py 训练模型')
