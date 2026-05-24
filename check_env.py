# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

# 检测 PyTorch
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
except ImportError:
    print('PyTorch: 未安装')

# 检测 CLIP
try:
    import clip
    print('CLIP: 已安装')
except ImportError:
    print('CLIP: 未安装')

# 检测 PIL
try:
    from PIL import Image
    print('Pillow: 已安装')
except ImportError:
    print('Pillow: 未安装')

# 检测 transformers
try:
    import transformers
    print(f'transformers: {transformers.__version__}')
except ImportError:
    print('transformers: 未安装')
