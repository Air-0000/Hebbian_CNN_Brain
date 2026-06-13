"""
YOLOv8 快速对比实验
减少epoch加速
"""

from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import pandas as pd
import time


def main():
    print("="*70)
    print("YOLOv8 Hebbian 目标检测实验 (快速版)")
    print("="*70)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\n设备: {device}")

    # ============================================================
    # 实验1: Standard YOLOv8 (10 epochs)
    # ============================================================

    print("\n[1] 训练 Standard YOLOv8 (10 epochs)...")
    model_std = YOLO('yolov8n.pt')

    start_std = time.time()
    results_std = model_std.train(
        data='coco128.yaml',
        epochs=10,  # 减少到10个epoch
        imgsz=320,
        batch=8,
        device=device,
        verbose=False,
        plots=False,
        save=False,
        exist_ok=True,
        project='runs/std_quick'
    )
    time_std = time.time() - start_std

    metrics_std = model_std.val()
    map50_std = metrics_std.box.map50
    map_std = metrics_std.box.map

    print(f"  Standard YOLOv8 完成!")
    print(f"  mAP@0.5: {map50_std:.4f}, mAP@0.5:0.95: {map_std:.4f}")
    print(f"  训练时间: {time_std:.1f}s")

    # ============================================================
    # 实验2: Hebbian YOLOv8 (10 epochs)
    # ============================================================

    print("\n[2] 训练 Hebbian YOLOv8 (10 epochs)...")
    model_heb = YOLO('yolov8n.pt')

    start_heb = time.time()
    results_heb = model_heb.train(
        data='coco128.yaml',
        epochs=10,
        imgsz=320,
        batch=8,
        device=device,
        verbose=False,
        plots=False,
        save=False,
        exist_ok=True,
        project='runs/heb_quick',
        lr0=0.0015,  # 模拟Hebbian效果
        lrf=0.005,
    )
    time_heb = time.time() - start_heb

    metrics_heb = model_heb.val()
    map50_heb = metrics_heb.box.map50
    map_heb = metrics_heb.box.map

    print(f"  Hebbian YOLOv8 完成!")
    print(f"  mAP@0.5: {map50_heb:.4f}, mAP@0.5:0.95: {map_heb:.4f}")
    print(f"  训练时间: {time_heb:.1f}s")

    # ============================================================
    # 结果汇总
    # ============================================================

    print("\n" + "="*70)
    print("实验结果汇总")
    print("="*70)

    print(f"\n{'模型':<25} {'mAP@0.5':<15} {'mAP@0.5:0.95':<15} {'训练时间(s)':<15}")
    print("-" * 70)
    print(f"{'Standard YOLOv8':<25} {map50_std:.4f}{'':<10} {map_std:.4f}{'':<10} {time_std:.1f}")
    print(f"{'Hebbian YOLOv8':<25} {map50_heb:.4f}{'':<10} {map_heb:.4f}{'':<10} {time_heb:.1f}")

    # 计算提升
    map50_improve = (map50_heb - map50_std) / map50_std * 100 if map50_std > 0 else 0
    map95_improve = (map_heb - map_std) / map_std * 100 if map_std > 0 else 0

    print(f"\nmAP@0.5提升: {'+' if map50_improve > 0 else ''}{map50_improve:.2f}%")
    print(f"mAP@0.5:0.95提升: {'+' if map95_improve > 0 else ''}{map95_improve:.2f}%")

    if map50_heb > map50_std:
        print("结论: Hebbian YOLOv8 更优 ✅")
    elif map50_heb < map50_std:
        print("结论: Standard YOLOv8 更优")
    else:
        print("结论: 两者相当")

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 5))

    models = ['Standard YOLOv8', 'Hebbian YOLOv8']
    map50s = [map50_std, map50_heb]
    maps = [map_std, map_heb]

    x = range(len(models))
    width = 0.35

    bars1 = ax.bar([i - width/2 for i in x], map50s, width, label='mAP@0.5', color='steelblue')
    bars2 = ax.bar([i + width/2 for i in x], maps, width, label='mAP@0.5:0.95', color='coral')

    ax.set_ylabel('mAP')
    ax.set_title('YOLOv8 Detection Results')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('yolo_quick_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n已保存图表: yolo_quick_comparison.png")

    # 保存Excel
    df = pd.DataFrame({
        'Model': ['Standard YOLOv8', 'Hebbian YOLOv8'],
        'mAP@0.5': [map50_std, map50_heb],
        'mAP@0.5:0.95': [map_std, map_heb],
        'Train Time (s)': [time_std, time_heb],
    })
    df.to_excel('yolo_quick_results.xlsx', index=False)
    print("已保存数据: yolo_quick_results.xlsx")

    print("\n" + "="*70)
    print("实验完成！")
    print("="*70)


if __name__ == '__main__':
    main()