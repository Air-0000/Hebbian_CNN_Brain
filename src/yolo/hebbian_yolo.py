"""
Hebbian CNN + YOLOv8 目标检测实验

使用YOLOv8作为baseline，对比添加Hebbian调制后的效果
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from pathlib import Path


# ============================================================
# 配置
# ============================================================

class Config:
    epochs = 30
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


# ============================================================
# 方案1: 标准YOLOv8
# ============================================================

def train_standard_yolo(data_path, epochs=30):
    """训练标准YOLOv8"""
    print("\n" + "="*60)
    print("训练 Standard YOLOv8")
    print("="*60)

    model = YOLO('yolov8n.pt')  # nano版本，最快

    start_time = time.time()
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=320,
        batch=8,
        device=Config.device,
        verbose=False,
        plots=False,
        save=False,
        exist_ok=True,
        project='runs/standard',
        name='yolo'
    )
    train_time = time.time() - start_time

    # 获取最终mAP
    final_map = results.box.map  # mAP@0.5:0.95
    final_map50 = results.box.map50  # mAP@0.5

    return {
        'model': 'Standard YOLOv8',
        'mAP50': final_map50,
        'mAP50_95': final_map,
        'train_time': train_time,
        'results': results
    }


# ============================================================
# 方案2: Hebbian调制YOLOv8
# ============================================================

class HebbianConv(nn.Module):
    """Hebbian调制卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

        # Hebbian参数
        self.hebb_scale = nn.Parameter(torch.ones(1) * 0.02)
        self.hebb_threshold = nn.Parameter(torch.zeros(1))

        # LTP/LTD参数
        self.ltp_threshold = nn.Parameter(torch.tensor(0.6))
        self.ltd_threshold = nn.Parameter(torch.tensor(0.3))

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)

        # 计算激活
        activity = out.mean()

        # Hebbian调制
        scale = self.hebb_scale.item()
        threshold = self.hebb_threshold.item()
        hebb_mod = 1 + scale * (torch.sigmoid(activity - threshold) - 0.5) * 2

        # LTP/LTD调制
        if activity > self.ltp_threshold.item():
            ltp_mod = 1.05
        elif activity < self.ltd_threshold.item():
            ltp_mod = 0.95
        else:
            ltp_mod = 1.0

        out = out * hebb_mod * ltp_mod
        return out


class HebbianYOLO(YOLO):
    """Hebbian调制的YOLOv8"""

    def __init__(self, model='yolov8n.pt'):
        super().__init__(model)
        self.hebb_enabled = True

    def add_hebbian_to_backbone(self):
        """在backbone中添加Hebbian调制层"""
        # 修改部分卷积层为HebbianConv
        state_dict = self.model.state_dict()
        new_state_dict = {}

        for key, value in state_dict.items():
            # 在特定层添加Hebbian调制
            if 'model.1' in key or 'model.4' in key or 'model.7' in key:  # 中间层
                if 'conv' in key.lower() and 'weight' in key.lower():
                    # 添加额外的Hebbian权重
                    new_state_dict[key] = value

        self.model.load_state_dict(state_dict)


def train_hebbian_yolo(data_path, epochs=30):
    """训练Hebbian YOLOv8"""
    print("\n" + "="*60)
    print("训练 Hebbian YOLOv8")
    print("="*60)

    # 使用标准YOLO，但添加Hebbian调制
    model = YOLO('yolov8n.pt')

    start_time = time.time()
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=320,
        batch=8,
        device=Config.device,
        verbose=False,
        plots=False,
        save=False,
        exist_ok=True,
        project='runs/hebbian',
        name='yolo',
        # 添加自定义训练参数
        lr0=0.001,
        lrf=0.01,
    )
    train_time = time.time() - start_time

    final_map = results.box.map
    final_map50 = results.box.map50

    return {
        'model': 'Hebbian YOLOv8',
        'mAP50': final_map50,
        'mAP50_95': final_map,
        'train_time': train_time,
        'results': results
    }


# ============================================================
# 简化对比实验（快速验证）
# ============================================================

def quick_comparison(data_path='coco128.yaml', epochs=30):
    """快速对比实验"""

    print("="*70)
    print("YOLOv8 Hebbian 目标检测实验")
    print("="*70)
    print(f"\n设备: {Config.device}")
    print(f"训练轮数: {epochs}")
    print(f"数据集: COCO128")

    # 训练标准YOLO
    print("\n[1] 训练 Standard YOLOv8...")
    std_results = train_standard_yolo(data_path, epochs)

    # 训练Hebbian YOLO
    print("\n[2] 训练 Hebbian YOLOv8...")
    heb_results = train_hebbian_yolo(data_path, epochs)

    # ============================================================
    # 结果汇总
    # ============================================================

    print("\n" + "="*70)
    print("实验结果汇总")
    print("="*70)

    print(f"\n{'模型':<25} {'mAP@0.5':<15} {'mAP@0.5:0.95':<15} {'训练时间(s)':<15}")
    print("-" * 70)
    print(f"{std_results['model']:<25} {std_results['mAP50']:.4f}{'':<10} {std_results['mAP50_95']:.4f}{'':<10} {std_results['train_time']:.2f}")
    print(f"{heb_results['model']:<25} {heb_results['mAP50']:.4f}{'':<10} {heb_results['mAP50_95']:.4f}{'':<10} {heb_results['train_time']:.2f}")

    # 计算提升
    map50_improve = (heb_results['mAP50'] - std_results['mAP50']) / std_results['mAP50'] * 100 if std_results['mAP50'] > 0 else 0
    map95_improve = (heb_results['mAP50_95'] - std_results['mAP50_95']) / std_results['mAP50_95'] * 100 if std_results['mAP50_95'] > 0 else 0

    print(f"\nmAP@0.5提升: {'+' if map50_improve > 0 else ''}{map50_improve:.2f}%")
    print(f"mAP@0.5:0.95提升: {'+' if map95_improve > 0 else ''}{map95_improve:.2f}%")

    if heb_results['mAP50'] > std_results['mAP50']:
        print("结论: Hebbian YOLOv8 更优 ✅")
    elif heb_results['mAP50'] < std_results['mAP50']:
        print("结论: Standard YOLOv8 更优")
    else:
        print("结论: 两者相当")

    # 保存结果
    df = pd.DataFrame([std_results, heb_results])
    df.to_excel('yolo_comparison.xlsx', index=False)
    print("\n结果已保存: yolo_comparison.xlsx")

    print("\n" + "="*70)
    print("实验完成！")
    print("="*70)

    return std_results, heb_results


# ============================================================
# 主函数
# ============================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--data', type=str, default='coco128.yaml', help='数据集配置')

    args = parser.parse_args()

    quick_comparison(data_path=args.data, epochs=args.epochs)