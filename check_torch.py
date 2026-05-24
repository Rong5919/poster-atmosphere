# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
except ImportError:
    print('PyTorch: 未安装')
