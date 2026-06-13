"""
自动化训练流程
1. 训练 BioHebbianNet v3 (93%版本) → 保存 pth
2. 检查结果 → 如果准确率接近93%
3. 自动训练完整全量 LoRA 版本
"""

import subprocess
import time
import os
import torch

print("="*70, flush=True)
print("自动化训练流程", flush=True)
print("="*70, flush=True)

# 步骤1: 训练93%版本
print("\n" + "="*70, flush=True)
print("步骤1: 训练 BioHebbianNet v3 (93%版本)", flush=True)
print("="*70, flush=True)

result = subprocess.run(
    ['python3', '-u', 'src/train/cifar/bio_hebbian_v3_final.py'],
    capture_output=False,
    cwd='/Users/constant/Desktop/Agent/AgentTeam/projects/Hebbian_CNN_Brain'
)

# 检查是否成功
pth_path = 'results/model_checkpoints/bio_hebbian_v3_best.pth'
if os.path.exists(pth_path):
    print(f"\n✓ 模型已保存: {pth_path}", flush=True)

    # 加载检查准确率
    checkpoint = torch.load(pth_path, weights_only=False)
    acc = checkpoint.get('best_acc', 0)
    params = checkpoint.get('params', 0)
    print(f"✓ 准确率: {acc:.2f}%", flush=True)
    print(f"✓ 参数量: {params:,} ({params/1e6:.2f}M)", flush=True)

    # 判断是否继续
    if acc >= 90.0:
        print(f"\n准确率 {acc:.2f}% >= 90%, 继续步骤2...", flush=True)

        print("\n" + "="*70, flush=True)
        print("步骤2: 训练完整全量 LoRA 版本", flush=True)
        print("="*70, flush=True)

        result = subprocess.run(
            ['python3', '-u', 'src/train/cifar/lora_bio_hebbian_v3_full.py'],
            capture_output=False,
            cwd='/Users/constant/Desktop/Agent/AgentTeam/projects/Hebbian_CNN_Brain'
        )

        print("\n" + "="*70, flush=True)
        print("全部训练完成!", flush=True)
        print("="*70, flush=True)
    else:
        print(f"\n准确率 {acc:.2f}% < 90%, 停止训练", flush=True)
else:
    print("\n✗ 训练失败或模型未保存", flush=True)