# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
import pandas as pd
import os
from torchvision import transforms
from PIL import Image
import numpy as np
from model import SimpleCNN

MODEL_PATH = r'C:\Users\admin\Desktop\RepViT\outputs\best_model.pth'
IMG_BASE = r'C:\Users\admin\Desktop\RepViT\dataset\posters_resize'

ATMOSPHERE_MAP = {
    0: "ke_ji_gan",   # 科技感
    1: "shi_shang_gan",  # 时尚感
    2: "xi_qing_gan",  # 喜庆感
    3: "jian_yue_shang_wu",  # 简约商务
    4: "fu_gu_guo_feng",  # 复古国风
    5: "ka_tong_tong_qu",  # 卡通童趣
    6: "gao_duan_qing_she"  # 高端轻奢
}

LABEL_TO_ID = {
    "科技感": 0, "时尚感": 1, "喜庆感": 2, "简约商务": 3,
    "复古国风": 4, "卡通童趣": 5, "高端轻奢": 6
}

train_df = pd.read_csv(r'C:\Users\admin\Desktop\RepViT\dataset\train.csv')
print(f'Train samples: {len(train_df)}')
print(f'Class distribution:')
print(train_df['atmosphere'].value_counts())

device = torch.device('cpu')
model = SimpleCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_color_stats(img):
    arr = np.array(img.resize((64, 64))).astype(float) / 255.0
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    brightness = (r*0.299 + g*0.587 + b*0.114).mean()
    # Saturation
    gray = r*0.299 + g*0.587 + b*0.114
    saturation = np.sqrt(((r-gray)**2 + (g-gray)**2 + (b-gray)**2).mean())
    warmth = r.mean() - b.mean()
    return brightness, saturation, r.mean(), g.mean(), b.mean(), warmth

def find_image(name, base):
    for subdir in os.listdir(base):
        candidate = os.path.join(base, subdir, name)
        if os.path.exists(candidate):
            return candidate
    return None

print('\n=== Class color statistics ===')
print(f'{"Class":12s} | {"Bright":6s} | {"Sat":6s} | {"R":6s} | {"G":6s} | {"B":6s} | {"Warm":6s} | N')
print('-' * 75)

for class_name in ["科技感", "时尚感", "喜庆感", "简约商务", "复古国风", "卡通童趣", "高端轻奢"]:
    samples = train_df[train_df['atmosphere'] == class_name]
    n = min(10, len(samples))
    stats_list = []
    for _, row in samples.head(n).iterrows():
        path = find_image(row['name'], IMG_BASE)
        if path:
            try:
                img = Image.open(path).convert('RGB')
                stats_list.append(get_color_stats(img))
            except:
                pass
    
    if stats_list:
        b = np.mean([s[0] for s in stats_list])
        s = np.mean([s[1] for s in stats_list])
        r = np.mean([s[2] for s in stats_list])
        g = np.mean([s[3] for s in stats_list])
        bl = np.mean([s[4] for s in stats_list])
        w = np.mean([s[5] for s in stats_list])
        print(f'{class_name:12s} | {b:.3f} | {s:.3f} | {r:.3f} | {g:.3f} | {bl:.3f} | {w:+.3f} | {len(stats_list)}')

print('\n=== Key observations ===')
print('warmth: positive=warm(red/yellow), negative=cool(blue)')
print('saturation: higher=more vivid colors')
print('brightness: higher=lighter')
