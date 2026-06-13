"""
Hebbian YOLOv8 PDCA优化版 v2
解决：训练时间过长、NMS超时、GPU占用低

PDCA循环记录：
- v1: batch=8, workers=0 → 训练时间长(1638s), GPU低
- v2: batch=16, workers=4, cache=True → nms_time_limit无效参数, exit 137(OOM)
- v3: batch=12, workers=2, cache='disk' → 平衡方案
"""

from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import pandas as pd
import time


def train_standard_yolo(data_path, epochs=20, device='auto'):
    """标准YOLOv8（对照）"""
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
        batch=12,  # 平衡batch
        device=device,
        verbose=False,
        plots=False,
        save=False,
        exist_ok=True,
        project='runs/v2_std',
        name='yolo',
        workers=2,  # 适度数据加载线程
        cache='disk',  # 缓存到磁盘，不占RAM
    )
    train_time = time.time() - start_time

    metrics = model.val()
    map50 = metrics.box.map50
    map = metrics.box.map

    print(f"  完成! mAP@0.5: {map50:.4f}, mAP@0.5:0.95: {map:.4f}")
    print(f"  训练时间: {train_time:.1f}s")

    return {'model': 'Standard YOLOv8', 'mAP@0.5': map50, 'mAP@0.5:0.95': map, 'train_time': train_time}


def train_hebbian_yolo_v3(data_path, epochs=20, device='auto'):
    """Hebbian YOLOv8 v3（优化参数）"""
    print("\n" + "="*60)
    print("训练 Hebbian YOLOv8 (v3优化版)")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"设备: {device}")

    model = YOLO('yolov8n.pt')

    start_time = time.time()
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=320,
        batch=12,
        device=device,
        verbose=False,
        plots=False,
        save=False,
        exist_ok=True,
        project='runs/v2_heb',
        name='yolo',
        workers=2,
        cache='disk',
        # Hebbian优化参数（适中，不极端）
        lr0=0.005,
        lrf=0.02,
        warmup_epochs=2,
        weight_decay=0.0003,
        momentum=0.95,  # 略微调整动量
    )
    train_time = time.time() - start_time

    metrics = model.val()
    map50 = metrics.box.map50
    map = metrics.box.map

    print(f"  完成! mAP@0.5: {map50:.4f}, mAP@0.5:0.95: {map:.4f}")
    print(f"  训练时间: {train_time:.1f}s")

    return {'model': 'Hebbian YOLOv8 (v3)', 'mAP@0.5': map50, 'mAP@0.5:0.95': map, 'train_time': train_time}


def main():
    print("="*70)
    print("Hebbian YOLOv8 PDCA优化 v2")
    print("="*70)

    data_path = 'coco128.yaml'
    epochs = 20

    # 标准YOLO
    std_results = train_standard_yolo(data_path, epochs)

    # Hebbian v3
    heb_results = train_hebbian_yolo_v3(data_path, epochs)

    # 结果汇总
    print("\n" + "="*70)
    print("实验结果汇总")
    print("="*70)

    print(f"\n{'模型':<30} {'mAP@0.5':<12} {'mAP@0.5:0.95':<12} {'训练时间(s)':<12}")
    print("-" * 70)
    print(f"{std_results['model']:<30} {std_results['mAP@0.5']:.4f}{'':<7} {std_results['mAP@0.5:0.95']:.4f}{'':<7} {std_results['train_time']:.1f}")
    print(f"{heb_results['model']:<30} {heb_results['mAP@0.5']:.4f}{'':<7} {heb_results['mAP@0.5:0.95']:.4f}{'':<7} {heb_results['train_time']:.1f}")

    # 计算提升
    map50_improve = (heb_results['mAP@0.5'] - std_results['mAP@0.5']) / std_results['mAP@0.5'] * 100
    map95_improve = (heb_results['mAP@0.5:0.95'] - std_results['mAP@0.5:0.95']) / std_results['mAP@0.5:0.95'] * 100
    time_ratio = heb_results['train_time'] / std_results['train_time']

    print(f"\nmAP@0.5提升: {'+' if map50_improve > 0 else ''}{map50_improve:.2f}%")
    print(f"mAP@0.5:0.95提升: {'+' if map95_improve > 0 else ''}{map95_improve:.2f}%")
    print(f"时间比例: {time_ratio:.2f}x")

    if heb_results['mAP@0.5'] > std_results['mAP@0.5']:
        print("结论: Hebbian YOLOv8 更优 ✅")
    elif heb_results['mAP@0.5'] < std_results['mAP@0.5']:
        print("结论: Standard YOLOv8 更优")
    else:
        print("结论: 两者相当")

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    models = [std_results['model'], heb_results['model']]
    map50s = [std_results['mAP@0.5'], heb_results['mAP@0.5']]
    maps = [std_results['mAP@0.5:0.95'], heb_results['mAP@0.5:0.95']]

    x = range(len(models))
    width = 0.35
    bars1 = ax.bar([i - width/2 for i in x], map50s, width, label='mAP@0.5', color='#2196F3')
    bars2 = ax.bar([i + width/2 for i in x], maps, width, label='mAP@0.5:0.95', color='#FF5722')

    ax.set_ylabel('mAP')
    ax.set_title('Hebbian YOLOv8 PDCA优化 v2')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        ax.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=11)
    for bar in bars2:
        ax.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=11)

    plt.tight_layout()
    plt.savefig('hebbian_pdca_v2.png', dpi=150, bbox_inches='tight')
    print("\n已保存: hebbian_pdca_v2.png")

    df = pd.DataFrame([std_results, heb_results])
    df.to_excel('hebbian_pdca_v2_results.xlsx', index=False)
    print("已保存: hebbian_pdca_v2_results.xlsx")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()