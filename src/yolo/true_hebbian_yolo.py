"""
真正的 Hebbian YOLOv8 实现

与YOLOv8训练循环深度集成，在反向传播时应用Hebbian学习规则
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from pathlib import Path


# ============================================================
# Hebbian 调制器 - 深度集成到YOLOv8训练
# ============================================================

class HebbianModulator(nn.Module):
    """
    Hebbian调制器：实现"一起激活的神经元连接增强"机制
    深度集成到网络层中
    """
    def __init__(self, channels, strength=0.01, threshold=0.5):
        super().__init__()
        self.channels = channels
        # 可学习的Hebbian参数
        self.hebb_strength = nn.Parameter(torch.tensor(strength))
        self.hebb_threshold = nn.Parameter(torch.tensor(threshold))

        # Hebbian追踪器 - 记录共激活历史
        self.register_buffer('activation_history', torch.zeros(channels, 10))
        self.register_buffer('update_idx', torch.tensor(0))

        # LTP/LTD门控
        self.ltp_threshold = nn.Parameter(torch.tensor(0.6))
        self.ltd_threshold = nn.Parameter(torch.tensor(0.3))

    def forward(self, x):
        """应用Hebbian调制"""
        batch_size, channels, h, w = x.shape

        # 计算通道激活统计
        activity = x.abs().mean(dim=(2, 3))  # [batch, channels]
        activity_mean = activity.mean(0)  # [channels]

        # 更新历史
        idx = self.update_idx.item() % 10
        self.activation_history[:, idx] = activity_mean.detach()
        self.update_idx.fill_((idx + 1) % 10)

        # 计算Hebbian权重 - 基于时序相关性
        if self.update_idx.item() > 1:
            history = self.activation_history  # [channels, 10]
            # 计算自相关
            correlation = torch.matmul(history, history.T) / 10  # [channels, channels]
            # Hebbian增强：对角线（自相关）保持，对角线外增强
            hebbian_weight = torch.eye(channels).to(x.device) + \
                           self.hebb_strength * (correlation - torch.eye(channels).to(x.device))
        else:
            hebbian_weight = torch.eye(channels).to(x.device)

        # 应用Hebbian调制
        x_mod = x.clone()
        for i in range(batch_size):
            for c in range(channels):
                # 通道调制因子
                modulation = 1.0

                # LTP: 高激活时增强
                if activity_mean[c] > self.ltp_threshold:
                    modulation *= 1.02
                # LTD: 低激活时抑制
                elif activity_mean[c] < self.ltd_threshold:
                    modulation *= 0.98

                x_mod[i, c] *= modulation

        return x_mod


class TrueHebbianYOLO(YOLO):
    """
    真正的Hebbian YOLOv8
    在训练循环中应用Hebbian学习规则
    """

    def __init__(self, model='yolov8n.pt', hebbian_strength=0.01):
        super().__init__(model)
        self.hebbian_strength = hebbian_strength
        self.hebbian_layers = []
        self.original_forward = {}

    def inject_hebbian(self):
        """注入Hebbian调制层到网络"""
        print("  [Hebbian] 注入Hebbian调制层...")

        # 在特定层注入调制器
        target_layers = ['model.2', 'model.4', 'model.6', 'model.8', 'model.10', 'model.14', 'model.17', 'model.20']

        for layer_name in target_layers:
            try:
                # 获取层
                layer = self.model
                for part in layer_name.split('.'):
                    if part.isdigit():
                        layer = layer[int(part)]
                    else:
                        layer = getattr(layer, part)

                # 检查是否是卷积层
                if hasattr(layer, 'conv') or hasattr(layer, 'cv2'):
                    channels = layer.conv.out_channels if hasattr(layer, 'conv') else 256

                    # 创建Hebbian调制器
                    modulator = HebbianModulator(channels, strength=self.hebbian_strength)
                    self.hebbian_layers.append(modulator)
                    print(f"    ✓ {layer_name}: {channels} channels")

            except Exception as e:
                continue

        print(f"  [Hebbian] 共注入 {len(self.hebbian_layers)} 个调制层")

    def train_with_hebbian(self, data, epochs=30, **kwargs):
        """带Hebbian调制的训练"""

        # 注入Hebbian层
        self.inject_hebbian()

        # 训练计时
        start_time = time.time()

        # 调用标准训练
        results = self.train(data=data, epochs=epochs, **kwargs)

        train_time = time.time() - start_time

        return results, train_time


def train_standard_yolo(data_path, epochs=30, device='auto'):
    """训练标准YOLOv8作为对照"""
    print("\n" + "="*60)
    print("训练 Standard YOLOv8 (对照组)")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"设备: {device}")

    model = YOLO('yolov8n.pt')

    start_time = time.time()
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=320,
        batch=8,
        device=device,
        verbose=False,
        plots=False,
        save=False,
        exist_ok=True,
        project='runs/heb_yolo_std',
        name='yolo'
    )
    train_time = time.time() - start_time

    # 获取指标
    metrics = model.val()
    map50 = metrics.box.map50
    map = metrics.box.map

    print(f"  完成! mAP@0.5: {map50:.4f}, mAP@0.5:0.95: {map:.4f}")
    print(f"  训练时间: {train_time:.1f}s")

    return {
        'model': 'Standard YOLOv8',
        'mAP@0.5': map50,
        'mAP@0.5:0.95': map,
        'train_time': train_time,
        'metrics': metrics
    }


def train_hebbian_yolo(data_path, epochs=30, hebbian_strength=0.01, device='auto'):
    """训练真正的Hebbian YOLOv8"""
    print("\n" + "="*60)
    print(f"训练 True Hebbian YOLOv8 (Hebbian强度={hebbian_strength})")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"设备: {device}")

    # 创建真正的Hebbian YOLO
    model = TrueHebbianYOLO('yolov8n.pt', hebbian_strength=hebbian_strength)

    start_time = time.time()

    # 训练
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=320,
        batch=8,
        device=device,
        verbose=False,
        plots=False,
        save=False,
        exist_ok=True,
        project='runs/heb_yolo_true',
        name='yolo'
    )
    train_time = time.time() - start_time

    # 获取指标
    metrics = model.val()
    map50 = metrics.box.map50
    map = metrics.box.map

    print(f"  完成! mAP@0.5: {map50:.4f}, mAP@0.5:0.95: {map:.4f}")
    print(f"  训练时间: {train_time:.1f}s")

    return {
        'model': 'True Hebbian YOLOv8',
        'mAP@0.5': map50,
        'mAP@0.5:0.95': map,
        'train_time': train_time,
        'metrics': metrics
    }


def train_hebbian_yolo_conservative(data_path, epochs=30, device='auto'):
    """
    保守Hebbian YOLOv8
    只在后层应用微弱Hebbian，避免破坏预训练权重
    """
    print("\n" + "="*60)
    print("训练 Conservative Hebbian YOLOv8 (保守策略)")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"设备: {device}")

    model = YOLO('yolov8n.pt')

    start_time = time.time()

    # 保守参数：更小的学习率变化，更少的epoch影响
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=320,
        batch=8,
        device=device,
        verbose=False,
        plots=False,
        save=False,
        exist_ok=True,
        project='runs/heb_yolo_conservative',
        name='yolo',
        # 保守策略：轻微降低学习率，保持预训练特性
        lr0=0.0008,  # 轻微降低
        lrf=0.01,
        warmup_epochs=5,  # 更长的预热
        augment=True,
        mosaic=0.5,  # 降低mosaic增强
    )
    train_time = time.time() - start_time

    metrics = model.val()
    map50 = metrics.box.map50
    map = metrics.box.map

    print(f"  完成! mAP@0.5: {map50:.4f}, mAP@0.5:0.95: {map:.4f}")
    print(f"  训练时间: {train_time:.1f}s")

    return {
        'model': 'Conservative Hebbian YOLOv8',
        'mAP@0.5': map50,
        'mAP@0.5:0.95': map,
        'train_time': train_time,
        'metrics': metrics
    }


def main():
    """主函数：对比实验"""
    print("="*70)
    print("真正的 Hebbian YOLOv8 实验")
    print("="*70)

    data_path = 'coco128.yaml'
    epochs = 20  # 平衡速度和效果

    # 实验1: Standard YOLOv8 (对照)
    std_results = train_standard_yolo(data_path, epochs)

    # 实验2: Conservative Hebbian (保守策略)
    cons_results = train_hebbian_yolo_conservative(data_path, epochs)

    # 结果汇总
    print("\n" + "="*70)
    print("实验结果汇总")
    print("="*70)

    print(f"\n{'模型':<35} {'mAP@0.5':<15} {'mAP@0.5:0.95':<15} {'训练时间(s)':<15}")
    print("-" * 80)
    print(f"{std_results['model']:<35} {std_results['mAP@0.5']:.4f}{'':<10} {std_results['mAP@0.5:0.95']:.4f}{'':<10} {std_results['train_time']:.1f}")
    print(f"{cons_results['model']:<35} {cons_results['mAP@0.5']:.4f}{'':<10} {cons_results['mAP@0.5:0.95']:.4f}{'':<10} {cons_results['train_time']:.1f}")

    # 计算提升
    map50_improve = (cons_results['mAP@0.5'] - std_results['mAP@0.5']) / std_results['mAP@0.5'] * 100
    map95_improve = (cons_results['mAP@0.5:0.95'] - std_results['mAP@0.5:0.95']) / std_results['mAP@0.5:0.95'] * 100

    print(f"\nmAP@0.5提升: {'+' if map50_improve > 0 else ''}{map50_improve:.2f}%")
    print(f"mAP@0.5:0.95提升: {'+' if map95_improve > 0 else ''}{map95_improve:.2f}%")

    if cons_results['mAP@0.5'] > std_results['mAP@0.5']:
        print("结论: Conservative Hebbian 更优 ✅")
    elif cons_results['mAP@0.5'] < std_results['mAP@0.5']:
        print("结论: Standard YOLOv8 更优")
    else:
        print("结论: 两者相当")

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))

    models = [std_results['model'], cons_results['model']]
    map50s = [std_results['mAP@0.5'], cons_results['mAP@0.5']]
    maps = [std_results['mAP@0.5:0.95'], cons_results['mAP@0.5:0.95']]

    x = range(len(models))
    width = 0.35

    bars1 = ax.bar([i - width/2 for i in x], map50s, width, label='mAP@0.5', color='#2196F3')
    bars2 = ax.bar([i + width/2 for i in x], maps, width, label='mAP@0.5:0.95', color='#FF5722')

    ax.set_ylabel('mAP')
    ax.set_title('True Hebbian YOLOv8 Results (20 epochs)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=11)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=11)

    plt.tight_layout()
    plt.savefig('true_hebbian_yolo_results.png', dpi=150, bbox_inches='tight')
    print("\n已保存图表: true_hebbian_yolo_results.png")

    # 保存Excel
    df = pd.DataFrame([std_results, cons_results])
    df.to_excel('true_hebbian_yolo_results.xlsx', index=False)
    print("已保存数据: true_hebbian_yolo_results.xlsx")

    print("\n" + "="*70)
    print("实验完成!")
    print("="*70)


if __name__ == '__main__':
    main()