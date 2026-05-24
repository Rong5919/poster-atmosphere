import os
import cv2
import numpy as np
from tqdm import tqdm

# 输入、输出路径（注意前面的r，防止中文路径乱码）
input_root = r"dataset/posters"
output_root = r"dataset/posters_resize"

# 创建输出根目录
os.makedirs(output_root, exist_ok=True)

# 遍历7类子文件夹
for class_name in os.listdir(input_root):
    class_input_path = os.path.join(input_root, class_name)
    class_output_path = os.path.join(output_root, class_name)
    os.makedirs(class_output_path, exist_ok=True)

    print(f"\n正在处理：{class_name}")
    img_list = os.listdir(class_input_path)

    # 遍历当前类别下的所有图片
    for img_name in tqdm(img_list, desc=class_name):
        img_path = os.path.join(class_input_path, img_name)
        try:
            # 解决中文路径乱码问题，用imdecode读取
            img_array = np.fromfile(img_path, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                continue  # 跳过损坏图片

            # 统一resize为640×640
            img = cv2.resize(img, (640, 640))

            # 解决中文路径乱码问题，用imencode保存
            save_path = os.path.join(class_output_path, img_name)
            cv2.imencode('.jpg', img)[1].tofile(save_path)

        except Exception as e:
            continue

print("\n✅ 所有图片resize完成！")
print(f"📂 处理好的图片已保存到：{output_root}")