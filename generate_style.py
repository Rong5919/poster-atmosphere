# -*- coding: utf-8 -*-
"""
RepViT 引导的智能海报生成 - HSV色彩变换版
使用 numpy + colorsys 进行真正的色相/饱和度/亮度联合变换
"""

import os
import sys
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import pandas as pd
import random

sys.stdout.reconfigure(encoding='utf-8')

from model import SimpleCNN


# ==================== 氛围标签映射 ====================
ATMOSPHERE_MAP = {
    0: "科技感", 1: "时尚感", 2: "喜庆感", 3: "简约商务",
    4: "复古国风", 5: "卡通童趣", 6: "高端轻奢"
}
LABEL_TO_ID = {v: k for k, v in ATMOSPHERE_MAP.items()}


# ============================================================
# HSV色彩空间参数
# hue: 0.0=红, 0.17=黄, 0.33=绿, 0.5=青, 0.67=蓝, 0.83=品红
# ============================================================
ATMOSPHERE_STYLE = {
    # 真实颜色统计:
    # 科技感: dark=0.31, sat=0.18, cool(blue)=-0.17  <- 暗蓝
    # 时尚感: bright=0.64, sat=0.14, warm=+0.02     <- 与高端重叠
    # 喜庆感: medium=0.53, sat=0.33, warm=+0.27     <- 高饱和+暖
    # 简约商务: bright=0.67, sat=0.11, warm=+0.05   <- 极低饱和
    # 复古国风: dark=0.31, sat=0.12, warm=+0.06     <- 暗+暖棕
    # 卡通童趣: bright=0.66, sat=0.27, warm=+0.15   <- 样本少(29)
    # 高端轻奢: bright=0.74, sat=0.17, warm=+0.05   <- 最亮
    0: {  # 科技感: 暗调 + 蓝色色相
        "hue_shift": 0.60,       # 色相转到蓝色区域 (0.67=纯蓝)
        "saturation": 1.3,        # 适度饱和提升
        "brightness": 0.75,       # 压暗
        "contrast": 1.4,
        "sharpness": 1.4,
        "desc": "暗调蓝色科技"
    },
    1: {  # 时尚感: 亮调 + 强粉紫色 (强到无法被误判为简约商务)
        "hue_shift": 0.83,        # 色相转到品红 (0.83=品红)
        "saturation": 1.8,        # 极高饱和 -> 摆脱"简约"感
        "brightness": 1.05,        # 亮
        "contrast": 1.25,
        "sharpness": 1.4,
        "color_overlay": (255, 0, 128),   # 强粉红叠加
        "overlay_strength": 0.25,          # 强叠加
        "desc": "强粉紫色时尚"
    },
    2: {  # 喜庆感: 暖红 + 高饱和
        "hue_shift": 0.98,        # 色相转到红色 (0.0=红，0.98接近红色)
        "saturation": 1.6,        # 高饱和
        "brightness": 1.0,        # 正常亮度
        "contrast": 1.3,
        "sharpness": 1.3,
        "desc": "红金高饱和喜庆"
    },
    3: {  # 简约商务: 亮调 + 极低饱和
        "hue_shift": 0.0,
        "saturation": 0.20,      # 极低饱和 (真实0.11=最低)
        "brightness": 1.12,      # 提亮
        "contrast": 1.4,
        "sharpness": 1.1,
        "desc": "极低饱和白底商务"
    },
    4: {  # 复古国风: 暖黄 + 低饱和 + 暗
        "hue_shift": 0.12,       # 色相转到橙黄
        "saturation": 0.50,      # 低饱和
        "brightness": 0.72,      # 压暗
        "contrast": 1.05,
        "sharpness": 1.0,
        "sepia_strength": 0.50,   # 叠加棕褐滤镜
        "desc": "暗棕水墨国风"
    },
    5: {  # 卡通童趣: 强橙黄 + 极高饱和 -> 与简约商务(低sat)的差异最大化
        "hue_shift": 0.10,        # 橙黄色相
        "saturation": 2.0,        # 极高饱和 (简约商务=0.20, 高端=0.65)
        "brightness": 1.12,       # 亮
        "contrast": 1.3,
        "sharpness": 1.3,
        "color_overlay": (255, 100, 0),   # 强橙色叠加
        "overlay_strength": 0.20,
        "desc": "强橙黄高饱和卡通"
    },
    6: {  # 高端轻奢: 最亮 + 香槟金 + 适中饱和
        "hue_shift": 0.10,        # 香槟金/暖黄
        "saturation": 0.65,       # 适中饱和 (简约0.11=极低,高端要更高)
        "brightness": 1.18,       # 最亮
        "contrast": 1.4,
        "sharpness": 1.5,
        "desc": "亮调香槟轻奢"
    }
}


def load_model(model_path, device):
    print(f'Loading model: {model_path}')
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f'Model loaded. Device: {device}')
    return model


def evaluate(model, img_tensor, device):
    with torch.no_grad():
        pred_cls, pred_reg = model(img_tensor.to(device))
    pred_class = pred_cls.argmax(dim=1).item()
    pred_label = ATMOSPHERE_MAP[pred_class]
    pred_scores = pred_reg[0].cpu().tolist()
    return pred_class, pred_label, pred_scores


def preprocess_image(img, size=224):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)


def rgb_to_hsv(r, g, b):
    """RGB (0-1) -> HSV (h=0-1, s=0-1, v=0-1)"""
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    
    if mx == mn:
        h = 0.0
    elif mx == r:
        h = (g - b) / df + (6 if g < b else 0)
    elif mx == g:
        h = (b - r) / df + 2
    else:
        h = (r - g) / df + 4
    h /= 6.0
    
    s = 0.0 if mx == 0 else df / mx
    v = mx
    return h, s, v


def hsv_to_rgb(h, s, v):
    """HSV (h=0-1, s=0-1, v=0-1) -> RGB (0-1)"""
    if s == 0:
        return v, v, v
    
    h = h % 1.0
    i = int(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    
    i = i % 6
    if i == 0: return v, t, p
    elif i == 1: return q, v, p
    elif i == 2: return p, v, t
    elif i == 3: return p, q, v
    elif i == 4: return t, p, v
    else: return v, p, q


def apply_atmosphere_style(img, style_params, strength=1.0):
    """
    应用氛围风格到图片
    使用真正的 HSV 色彩空间变换
    """
    img = img.convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0
    s = strength
    
    h_img, w_img = arr.shape[:2]
    flat = arr.reshape(-1, 3)  # (N, 3)
    
    # ---- HSV 变换 ----
    hsv_list = [rgb_to_hsv(p[0], p[1], p[2]) for p in flat]
    h_ch = np.array([x[0] for x in hsv_list])
    sv_ch = np.array([x[1] for x in hsv_list])
    v_ch = np.array([x[2] for x in hsv_list])
    
    # 1. 色相偏移
    if "hue_shift" in style_params:
        shift = style_params["hue_shift"] * s
        h_ch = (h_ch + shift) % 1.0
    
    # 2. 饱和度缩放
    if "saturation" in style_params:
        sat_scale = style_params["saturation"]
        sat_actual = 1.0 + (sat_scale - 1.0) * s
        sv_ch = np.clip(sv_ch * sat_actual, 0, 1)
    
    # 3. 亮度缩放
    if "brightness" in style_params:
        bright_scale = style_params["brightness"]
        bright_actual = 1.0 + (bright_scale - 1.0) * s
        v_ch = np.clip(v_ch * bright_actual, 0, 1)
    
    # 4. 对比度 (亮度中心扩展)
    if "contrast" in style_params:
        c = style_params["contrast"]
        c_actual = 1.0 + (c - 1.0) * s
        v_ch = np.clip((v_ch - 0.5) * c_actual + 0.5, 0, 1)
    
    # HSV -> RGB
    rgb_flat = np.array([hsv_to_rgb(ha, sa, va) for ha, sa, va in zip(h_ch, sv_ch, v_ch)])
    rgb_arr = rgb_flat.reshape(h_img, w_img, 3)
    rgb_arr = np.clip(rgb_arr, 0, 1)
    
    result = Image.fromarray((rgb_arr * 255).astype(np.uint8))
    
    # ---- 后处理 ----
    # 颜色叠加
    if style_params.get("color_overlay") and style_params.get("overlay_strength", 0) > 0:
        overlay_color = style_params["color_overlay"]
        overlay_alpha = style_params["overlay_strength"] * s
        overlay = Image.new('RGB', result.size, overlay_color)
        overlay = ImageEnhance.Brightness(overlay).enhance(0.35)
        result = Image.blend(result, overlay, overlay_alpha)
    
    # 棕褐滤镜
    if "sepia_strength" in style_params:
        sepia = style_params["sepia_strength"] * s
        if sepia > 0:
            gray = ImageOps.grayscale(result)
            gray_arr = np.array(gray)
            r_ch = np.clip(gray_arr * 1.1 + 30, 0, 255).astype(np.uint8)
            g_ch = np.clip(gray_arr * 0.9 + 20, 0, 255).astype(np.uint8)
            b_ch = np.clip(gray_arr * 0.8, 0, 255).astype(np.uint8)
            sepia_arr = np.stack([r_ch, g_ch, b_ch], axis=-1)
            sepia_img = Image.fromarray(sepia_arr)
            result = Image.blend(result, sepia_img, sepia)
    
    # 锐化
    sharp = style_params.get("sharpness", 1.0)
    if sharp > 1.0:
        enhancer = ImageEnhance.Sharpness(result)
        result = enhancer.enhance(1.0 + (sharp - 1.0) * s)
    
    return result


def find_source_images(base_dir, atmosphere=None, count=5):
    """从子目录中随机选取源图"""
    subdirs = {}
    for name in os.listdir(base_dir):
        subdir = os.path.join(base_dir, name)
        if os.path.isdir(subdir):
            files = [f for f in os.listdir(subdir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            if files:
                subdirs[name] = files
    
    candidates = []
    for name, files in subdirs.items():
        if atmosphere and name != atmosphere:
            for f in files:
                candidates.append((os.path.join(base_dir, name, f), name))
        elif not atmosphere:
            for f in files:
                candidates.append((os.path.join(base_dir, name, f), name))
    
    if not candidates:
        for name, files in subdirs.items():
            for f in files[:3]:
                candidates.append((os.path.join(base_dir, name, f), name))
    
    random.shuffle(candidates)
    return candidates[:count]


def iterative_style_transfer(source_img, target_class, model, device, max_iter=3):
    """迭代风格迁移"""
    target_label = ATMOSPHERE_MAP[target_class]
    style = ATMOSPHERE_STYLE[target_class]
    
    print(f'  Target: {target_label} ({style["desc"]})')
    
    best_img = None
    best_score = -1
    best_pred_class = -1
    best_history = []
    
    strengths = [0.5, 0.75, 1.0]  # 从较弱的强度开始
    
    for i, strength in enumerate(strengths):
        styled = apply_atmosphere_style(source_img, style, strength=strength)
        tensor = preprocess_image(styled)
        pred_class, pred_label, pred_scores = evaluate(model, tensor, device)
        
        avg_score = sum(pred_scores) / len(pred_scores)
        is_match = (pred_class == target_class)
        score = avg_score + 2.0 if is_match else avg_score
        
        best_history.append({
            'iter': i+1, 'strength': strength,
            'pred_class': pred_class, 'pred_label': pred_label,
            'scores': pred_scores, 'avg': avg_score, 'match': is_match
        })
        
        print(f'    iter={i+1} str={strength:.2f} | pred={pred_label} {"OK" if is_match else "X"} '
              f'| score={avg_score:.2f}')
        
        if score > best_score:
            best_score = score
            best_img = styled
            best_pred_class = pred_class
        
        if is_match:
            print(f'    ==> Match found at strength={strength}')
            break
    
    return best_img, best_score, best_history


def generate_for_atmosphere(target_class, model, device, img_base, n_sources=5):
    """为指定氛围生成海报"""
    target_label = ATMOSPHERE_MAP[target_class]
    print(f'\n{"="*60}')
    print(f'GENERATE: {target_label}')
    print(f'{"="*60}')
    
    candidates = find_source_images(img_base, atmosphere=target_label, count=n_sources)
    if not candidates:
        print('No candidates found')
        return None, None
    
    print(f'Candidates: {len(candidates)}')
    
    best_img = None
    best_score = -1
    best_info = None
    
    for j, (src_path, src_atm) in enumerate(candidates):
        print(f'\n[Source {j+1}/{len(candidates)}] {src_atm}')
        
        try:
            source_img = Image.open(src_path).convert('RGB')
        except Exception as e:
            print(f'  Load failed: {e}')
            continue
        
        styled, score, history = iterative_style_transfer(
            source_img, target_class, model, device, max_iter=3
        )
        
        last = history[-1]
        if last['match'] or score > best_score:
            best_score = score
            best_img = styled
            best_info = {
                'source': src_path, 'source_atmosphere': src_atm,
                'history': history,
                'final_pred': last['pred_label'],
                'final_scores': last['scores']
            }
        
        if last['match']:
            print(f'  => Match found! Stop.')
            break
    
    return best_img, best_info


def batch_generate_all(model, device, img_base, output_dir='generated'):
    """批量生成所有7种氛围"""
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    for class_id in range(7):
        target_label = ATMOSPHERE_MAP[class_id]
        # 时尚感和卡通童趣样本少，多试一些源图
        n = 10 if class_id in (1, 5) else 5
        img, info = generate_for_atmosphere(class_id, model, device, img_base, n_sources=n)
        
        if img is not None:
            save_path = os.path.join(output_dir, f'{target_label}.png')
            img.save(save_path)
            results[class_id] = {'success': True, 'path': save_path, 'info': info}
            print(f'\nSaved: {save_path}')
            if info:
                print(f'  Source: {info["source_atmosphere"]}')
                print(f'  Final: {info["final_pred"]}')
        else:
            results[class_id] = {'success': False, 'path': None, 'info': None}
            print(f'\nFailed: {target_label}')
    
    # Summary
    print(f'\n{"="*60}')
    print('SUMMARY')
    print(f'{"="*60}')
    ok = sum(1 for r in results.values() if r['success'] and 
             r['info'] and r['info']['final_pred'] == ATMOSPHERE_MAP[r['success'] and list(results.keys())[[i for i in results if r['success']][0] if False else 0]])
    
    # Simpler summary
    for k, v in ATMOSPHERE_MAP.items():
        if results[k]['success']:
            info = results[k]['info']
            ok_sym = "OK" if info and info['final_pred'] == v else "~"
            print(f'  [{ok_sym}] {v} -> {info["final_pred"] if info else "?"}')
        else:
            print(f'  [X] {v}')
    
    return results


def main():
    MODEL_PATH = r'C:\Users\admin\Desktop\RepViT\outputs\best_model.pth'
    IMG_BASE = r'C:\Users\admin\Desktop\RepViT\dataset\posters_resize'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('='*60)
    print('RepViT-Guided Poster Generation (HSV Color Transform)')
    print('='*60)
    print(f'Device: {DEVICE}')
    
    model = load_model(MODEL_PATH, DEVICE)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='batch',
                       choices=['batch', 'single'])
    parser.add_argument('--target', type=int, default=None)
    args = parser.parse_args()
    
    if args.mode == 'batch':
        batch_generate_all(model, DEVICE, IMG_BASE)
    elif args.mode == 'single' and args.target is not None:
        img, info = generate_for_atmosphere(args.target, model, DEVICE, IMG_BASE, n_sources=5)
        if img is not None:
            os.makedirs('generated', exist_ok=True)
            save_path = f'generated/{ATMOSPHERE_MAP[args.target]}.png'
            img.save(save_path)
            print(f'Saved: {save_path}')


if __name__ == '__main__':
    main()
