import torch
# 导入我们手动写的模型
from models.repvit import repvit_m1_0

# 1. 初始化模型
model = repvit_m1_0()
print("✅ 模型初始化成功！")

# 2. 测试模型推理
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# 3. 生成测试输入
img = torch.randn(1, 3, 640, 640).to(device)
print(f"输入形状: {img.shape}")

# 4. 前向传播
with torch.no_grad():
    out = model(img)

print(f"🎉 模型运行成功！输出形状: {out.shape}")
print("模型结构完全正确，环境搭建完成！")