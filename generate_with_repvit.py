# -*- coding: utf-8 -*-
"""
第三阶段：RepViT 引导的智能海报生成
使用训练好的 SimpleCNN 模型引导 Stable Diffusion 生成目标氛围的海报
"""

import os
import sys
import torch
from torchvision import transforms
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

# 导入模型
from model import SimpleCNN


# ==================== 氛围标签映射 ====================
# 数字 -> 中文
ATMOSPHERE_MAP = {
    0: "科技感",
    1: "时尚感",
    2: "喜庆感",
    3: "简约商务",
    4: "复古国风",
    5: "卡通童趣",
    6: "高端轻奢"
}

# 中文 -> 数字
LABEL_TO_ID = {v: k for k, v in ATMOSPHERE_MAP.items()}

# 每个氛围对应的生成提示词模板
PROMPT_TEMPLATES = {
    0: "科技海报，深蓝色调，霓虹灯光，未来感，高清质感，数字风格",
    1: "时尚海报，亮色配色，潮流元素，现代感，高清质感，商业摄影",
    2: "喜庆海报，红色金色，节日氛围，热闘喜庆，高清质感，传统风格",
    3: "简约商务海报，白色背景，极简风格，干净利落，高清质感，专业设计",
    4: "复古国风海报，水墨画风格，中国传统元素，古典韵味，高清质感，艺术品",
    5: "卡通童趣海报，动漫风格，明亮色彩，活泼可爱，高清质感，儿童插画",
    6: "高端轻奢海报，香槟金色调，精致细节，高贵典雅，高清质感，奢侈品牌风格"
}


def load_score_model(model_path, device):
    """
    加载训练好的评价模型
    
    Args:
        model_path: 模型权重文件路径
        device: 计算设备
    
    Returns:
        model: 加载好的模型
    """
    print(f'\n加载评价模型: {model_path}')
    
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f'模型加载成功，使用设备: {device}')
    return model


def find_image_in_subdirs(image_name, base_dir):
    """在子目录中查找图片"""
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            candidate = os.path.join(subdir_path, image_name)
            if os.path.exists(candidate):
                return candidate
    return None


def evaluate_image(image_path, model, device):
    """
    评估单张图片的氛围和评分
    
    Args:
        image_path: 图片路径
        model: 评价模型
        device: 计算设备
    
    Returns:
        pred_class: 预测的氛围类别 (0-6)
        pred_label: 预测的氛围名称
        pred_scores: 预测的5个评分 [color, composition, style, emotion, business]
    """
    # 图片预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载并预处理图片
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        pred_cls, pred_reg = model(img_tensor)
    
    # 解析结果
    pred_class = pred_cls.argmax(dim=1).item()
    pred_label = ATMOSPHERE_MAP[pred_class]
    pred_scores = pred_reg[0].cpu().tolist()
    
    return pred_class, pred_label, pred_scores


def print_evaluation(image_name, pred_label, target_label, pred_scores, is_match):
    """打印评估结果"""
    print(f'\n图片: {image_name}')
    print(f'目标氛围: {target_label}')
    print(f'预测氛围: {pred_label} {"✅ 匹配!" if is_match else "❌ 不匹配"}')
    print(f'预测评分:')
    print(f'  颜色(color):       {pred_scores[0]:.2f}')
    print(f'  构图(composition): {pred_scores[1]:.2f}')
    print(f'  风格(style):       {pred_scores[2]:.2f}')
    print(f'  情感(emotion):     {pred_scores[3]:.2f}')
    print(f'  商业(business):    {pred_scores[4]:.2f}')


def generate_with_prompt(target_class, pipe=None, device='cpu', iteration=1):
    """
    使用提示词生成图片
    
    Args:
        target_class: 目标氛围类别 (0-6)
        pipe: Stable Diffusion pipeline (如果为None则跳过生成)
        device: 计算设备
        iteration: 迭代次数
    
    Returns:
        image: 生成的图片 (PIL Image)
    """
    if pipe is None:
        print('\n⚠️ Stable Diffusion 未初始化，跳过生成')
        return None
    
    target_label = ATMOSPHERE_MAP[target_class]
    prompt = PROMPT_TEMPLATES[target_class]
    negative_prompt = "低质量，模糊，变形，水印，文字过多，杂乱"
    
    print(f'\n正在生成 "{target_label}" 氛围海报...')
    print(f'提示词: {prompt}')
    
    try:
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=20,
                guidance_scale=7.5
            ).images[0]
        
        save_path = f'generated_{target_label}_{iteration}.png'
        image.save(save_path)
        print(f'✅ 图片已保存: {save_path}')
        
        return image
        
    except Exception as e:
        print(f'❌ 生成失败: {e}')
        return None


def iterative_generation(target_class, model, pipe, device, max_iterations=3):
    """
    迭代生成直到达到目标氛围
    
    Args:
        target_class: 目标氛围类别 (0-6)
        model: 评价模型
        pipe: Stable Diffusion pipeline
        device: 计算设备
        max_iterations: 最大迭代次数
    
    Returns:
        success: 是否成功生成匹配的图片
        best_image: 最终选择的图片
    """
    target_label = ATMOSPHERE_MAP[target_class]
    print(f'\n{"="*60}')
    print(f'开始 RepViT 引导生成: 目标氛围 = {target_label}')
    print(f'{"="*60}')
    
    best_image = None
    best_score = -1
    
    for i in range(max_iterations):
        print(f'\n>>> 第 {i+1}/{max_iterations} 次迭代')
        
        # 生成图片
        image = generate_with_prompt(target_class, pipe, device, i+1)
        if image is None:
            continue
        
        # 保存临时文件用于评估
        temp_path = f'temp_gen_{i+1}.png'
        image.save(temp_path)
        
        # 评估
        pred_class, pred_label, pred_scores = evaluate_image(temp_path, model, device)
        is_match = (pred_class == target_class)
        
        print_evaluation(f'temp_gen_{i+1}.png', pred_label, target_label, pred_scores, is_match)
        
        # 计算综合得分
        avg_score = sum(pred_scores) / len(pred_scores)
        if is_match and avg_score > best_score:
            best_score = avg_score
            best_image = image
        
        # 如果匹配则停止
        if is_match:
            print(f'\n🎉 第 {i+1} 次迭代达到目标氛围！')
            best_image = image
            break
    
    return best_image is not None, best_image


def batch_generate_all_atmospheres(model, pipe, device, max_iter=3):
    """为所有7种氛围生成海报"""
    print(f'\n\n{"="*60}')
    print('批量生成：所有7种氛围海报')
    print(f'{"="*60}')
    
    os.makedirs('generated', exist_ok=True)
    
    results = {}
    
    for class_id in range(7):
        target_label = ATMOSPHERE_MAP[class_id]
        print(f'\n\n{"#"*60}')
        print(f'# 氛围 {class_id+1}/7: {target_label}')
        print(f'{"#"*60}')
        
        success, image = iterative_generation(class_id, model, pipe, device, max_iter)
        
        if success and image:
            save_path = f'generated/poster_{target_label}.png'
            image.save(save_path)
            print(f'✅ 最终保存: {save_path}')
            results[class_id] = {'success': True, 'path': save_path}
        else:
            print(f'⚠️ 未能生成匹配的 {target_label} 海报')
            results[class_id] = {'success': False, 'path': None}
    
    # 汇总
    print(f'\n\n{"="*60}')
    print('批量生成汇总')
    print(f'{"="*60}')
    success_count = sum(1 for r in results.values() if r['success'])
    print(f'成功: {success_count}/7')
    for class_id, result in results.items():
        status = "✅" if result['success'] else "❌"
        print(f'  {status} {ATMOSPHERE_MAP[class_id]}')
    
    return results


def test_model_on_samples(model_path, test_csv, device):
    """在测试集上测试模型效果"""
    import pandas as pd
    
    print(f'\n\n{"="*60}')
    print('在测试集上评估模型')
    print(f'{"="*60}')
    
    IMG_BASE = r'C:\Users\admin\Desktop\RepViT\dataset\posters_resize'
    df = pd.read_csv(test_csv)
    model = load_score_model(model_path, device)
    
    correct = 0
    total = 0
    per_class_correct = {k: 0 for k in range(7)}
    per_class_total = {k: 0 for k in range(7)}
    
    for idx, row in df.iterrows():
        image_name = row['name']
        true_label = row['atmosphere']
        
        if isinstance(true_label, str):
            true_class = LABEL_TO_ID.get(true_label, -1)
        else:
            true_class = int(true_label)
        
        if true_class not in ATMOSPHERE_MAP:
            continue
        
        image_path = find_image_in_subdirs(image_name, IMG_BASE)
        if image_path is None:
            continue
        
        pred_class, pred_label, pred_scores = evaluate_image(image_path, model, device)
        is_match = (pred_class == true_class)
        correct += is_match
        total += 1
        per_class_total[true_class] += 1
        per_class_correct[true_class] += is_match
        
        if total <= 10:
            true_name = ATMOSPHERE_MAP[true_class]
            print(f'\n[{total}] 真实:{true_name:4s} | 预测:{pred_label:4s} | {"✅" if is_match else "❌"}')
            print(f'     评分: 色{pred_scores[0]:.1f} 构{pred_scores[1]:.1f} 风{pred_scores[2]:.1f} 情{pred_scores[3]:.1f} 商{pred_scores[4]:.1f}')
    
    accuracy = correct / total * 100 if total > 0 else 0
    print(f'\n\n{"="*60}')
    print(f'测试集准确率: {accuracy:.2f}% ({correct}/{total})')
    print(f'{"="*60}')
    print('\n各类别准确率:')
    for k, v in ATMOSPHERE_MAP.items():
        t = per_class_total[k]
        c = per_class_correct[k]
        acc = c/t*100 if t > 0 else 0
        print(f'  {v:4s}: {acc:5.1f}% ({c}/{t})')
    
    return accuracy


def main():
    """主函数"""
    # 配置
    MODEL_PATH = r'C:\Users\admin\Desktop\RepViT\outputs\best_model.pth'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('='*60)
    print('第三阶段：RepViT 引导的智能海报生成')
    print('='*60)
    print(f'\n设备: {DEVICE}')
    
    # 加载评价模型
    model = load_score_model(MODEL_PATH, DEVICE)
    
    # 尝试加载 Stable Diffusion
    pipe = None
    try:
        from diffusers import StableDiffusionPipeline
        print('\n正在加载 Stable Diffusion 模型...')
        print('(首次运行需要下载模型，可能需要几分钟)')
        
        pipe = StableDiffusionPipeline.from_pretrained(
            'runwayml/stable-diffusion-v1-5',
            torch_dtype=torch.float16 if DEVICE.type == 'cuda' else torch.float32
        )
        pipe = pipe.to(DEVICE)
        print('✅ Stable Diffusion 加载成功')
        
    except ImportError:
        print('\n⚠️ diffusers 库未安装')
        print('请运行: pip install diffusers accelerate transformers')
        print('\n可以使用 --test-only 参数只测试评价模型')
        
    except Exception as e:
        print(f'\n⚠️ Stable Diffusion 加载失败: {e}')
        print('继续使用纯评价模式...')
    
    # 命令行参数处理
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='menu', 
                       choices=['menu', 'test', 'generate', 'batch', 'test-only'],
                       help='运行模式')
    parser.add_argument('--target', type=int, default=None,
                       help='目标氛围类别 (0-6)')
    parser.add_argument('--image', type=str, default=None,
                       help='要评估的图片路径')
    args = parser.parse_args()
    
    # 测试模式
    if args.mode == 'test' or args.image:
        test_csv = r'C:\Users\admin\Desktop\RepViT\dataset\test.csv'
        test_model_on_samples(MODEL_PATH, test_csv, DEVICE)
        return
    
    # 单张图片评估
    if args.image:
        pred_class, pred_label, pred_scores = evaluate_image(args.image, model, DEVICE)
        is_match = (args.target is not None and pred_class == args.target)
        target_label = ATMOSPHERE_MAP[args.target] if args.target is not None else "未指定"
        print_evaluation(args.image, pred_label, target_label, pred_scores, is_match)
        return
    
    # 交互式菜单
    print('\n' + '='*60)
    print('选择操作:')
    print('='*60)
    print('1. 测试模型效果 (在测试集上评估)')
    print('2. 评估单张图片')
    print('3. 为指定氛围生成海报')
    print('4. 批量生成所有7种氛围海报')
    print('5. 退出')
    
    choice = input('\n请输入选择 (1-5): ').strip()
    
    if choice == '1':
        test_csv = r'C:\Users\admin\Desktop\RepViT\dataset\test.csv'
        test_model_on_samples(MODEL_PATH, test_csv, DEVICE)
        
    elif choice == '2':
        image_path = input('请输入图片路径: ').strip().strip('"')
        if os.path.exists(image_path):
            pred_class, pred_label, pred_scores = evaluate_image(image_path, model, DEVICE)
            print_evaluation(image_path, pred_label, "N/A", pred_scores, False)
        else:
            print(f'文件不存在: {image_path}')
            
    elif choice == '3':
        print('\n可用氛围:')
        for k, v in ATMOSPHERE_MAP.items():
            print(f'  {k}: {v}')
        target = int(input('选择目标氛围 (0-6): '))
        if 0 <= target <= 6:
            success, image = iterative_generation(target, model, pipe, DEVICE)
            if success:
                save_path = f'generated/poster_{ATMOSPHERE_MAP[target]}.png'
                os.makedirs('generated', exist_ok=True)
                image.save(save_path)
                print(f'✅ 保存至: {save_path}')
        else:
            print('无效的选择')
            
    elif choice == '4':
        if pipe is None:
            print('\n⚠️ Stable Diffusion 未加载，无法生成')
            print('请先安装: pip install diffusers accelerate transformers')
            return
        batch_generate_all_atmospheres(model, pipe, DEVICE)
        
    elif choice == '5':
        print('退出')
    else:
        print('无效选择')


if __name__ == '__main__':
    main()
