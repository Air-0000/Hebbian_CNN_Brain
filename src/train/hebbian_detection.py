"""
Hebbian CNN 目标检测实验

同时做检测+分类：
- 标准CNN vs Hebbian CNN
- 使用COCO128真实标注
- 评估mAP指标
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from pathlib import Path
from PIL import Image
from collections import defaultdict


# ============================================================
# 配置
# ============================================================

class Config:
    epochs = 30
    batch_size = 4
    n_classes = 80
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    img_size = 320
    conf_threshold = 0.3
    iou_threshold = 0.5


# ============================================================
# COCO128数据集（目标检测格式）
# ============================================================

class COCO128DetectionDataset(Dataset):
    """COCO128目标检测数据集"""

    classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    def __init__(self, data_dir, split='train', max_samples=128):
        self.data_dir = Path(data_dir)
        self.max_samples = max_samples
        self.img_size = Config.img_size

        split_dir = 'train2017' if split == 'train' else 'val2017'
        self.img_dir = self.data_dir / 'images' / split_dir
        self.label_dir = self.data_dir / 'labels' / split_dir

        self.img_files = sorted(
            list(self.img_dir.glob('*.jpg')) + list(self.img_dir.glob('*.png'))
        )[:max_samples]

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]

        # 加载图像
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        img_tensor = self.transform(img)

        # 解析标签（YOLO格式）
        label_path = self.label_dir / (img_path.stem + '.txt')
        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        x_c, y_c, w, h = map(float, parts[1:5])

                        # 转换为像素坐标
                        x1 = (x_c - w/2) * orig_w
                        y1 = (y_c - h/2) * orig_h
                        x2 = (x_c + w/2) * orig_w
                        y2 = (y_c + h/2) * orig_h

                        # 缩放到[0,1]
                        x1, x2 = x1/orig_w, x2/orig_w
                        y1, y2 = y1/orig_h, y2/orig_h

                        boxes.append([x1, y1, x2, y2])
                        labels.append(cls)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4))
            labels = torch.zeros((0,), dtype=torch.long)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)

        return img_tensor, boxes, labels, str(img_path)


def collate_fn(batch):
    """处理变长检测目标"""
    images, boxes, labels, paths = zip(*batch)
    return torch.stack(images), boxes, labels, paths


# ============================================================
# 检测网络
# ============================================================

class DetectionCNN(nn.Module):
    """标准CNN检测器"""
    def __init__(self, n_classes=80):
        super().__init__()
        self.n_classes = n_classes

        # Backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # 检测头：输出 [batch, 5+80, h, w]
        # 5 = [x, y, w, h, conf]
        self.head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 85, 1),  # 85 = 5(bbox+conf) + 80(classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        out = self.head(feat)  # [B, 85, H, W]
        return out


class HebbianDetectionCNN(nn.Module):
    """Hebbian CNN检测器"""
    def __init__(self, n_classes=80):
        super().__init__()
        self.n_classes = n_classes

        # Hebbian参数
        self.hebb_scale = nn.Parameter(torch.ones(5) * 0.03)
        self.hebb_threshold = nn.Parameter(torch.zeros(5))

        # Backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # 检测头
        self.head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 85, 1),
        )

    def forward(self, x):
        feat = self.backbone(x)

        # Hebbian调制各层 - 使用新张量避免in-place操作
        B, C, H, W = feat.shape
        chunk_size = C // 5

        # 计算调制因子
        mod_factors = []
        for i in range(5):
            start = i * chunk_size
            end = min((i+1) * chunk_size, C)
            if start < C:
                layer_feat = feat[:, start:end]
                activity = layer_feat.mean().item()
                scale = self.hebb_scale[i].item()
                threshold = self.hebb_threshold[i].item()
                modulation = scale * (1 / (1 + np.exp(-(activity - threshold))) - 0.5) * 2
                mod_factors.append(1 + modulation)
            else:
                mod_factors.append(1.0)

        # 应用调制
        feat_new = feat.clone()
        for i in range(5):
            start = i * chunk_size
            end = min((i+1) * chunk_size, C)
            if start < C:
                feat_new[:, start:end] = feat[:, start:end] * mod_factors[i]

        out = self.head(feat_new)
        return out


# ============================================================
# 简化检测输出（用于评估）
# ============================================================

def parse_detection_output(pred, conf_thresh=0.3):
    """
    解析检测输出
    pred: [85, H, W] -> [5+80, H, W]
    返回: [(bbox, class, conf), ...]
    """
    # pred shape: [85, H, W]
    conf_map = pred[4]  # 置信度
    class_maps = pred[5:]  # 80个类的概率

    H, W = conf_map.shape
    boxes = []
    confs = []
    classes = []

    for i in range(H):
        for j in range(W):
            conf = conf_map[i, j].item()
            if conf > conf_thresh:
                # 找到最大概率的类
                class_probs = class_maps[:, i, j]
                class_id = class_probs.argmax().item()
                class_conf = class_probs[class_id].item()

                # 简单生成框（基于grid位置）
                x_center = (j + 0.5) / W
                y_center = (i + 0.5) / H
                box_w = 0.1
                box_h = 0.1

                x1 = max(0, x_center - box_w/2)
                y1 = max(0, y_center - box_h/2)
                x2 = min(1, x_center + box_w/2)
                y2 = min(1, y_center + box_h/2)

                boxes.append([x1, y1, x2, y2])
                confs.append(conf * class_conf)
                classes.append(class_id)

    return boxes, classes, confs


def compute_ap(precision, recall):
    """计算AP"""
    recall = np.concatenate(([0.], recall, [1.]))
    precision = np.concatenate(([0.], precision, [0.]))

    for i in range(precision.size - 1, 0, -1):
        precision[i-1] = np.maximum(precision[i-1], precision[i])

    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])

    return ap


def compute_map(predictions, ground_truths, iou_threshold=0.5, n_classes=80):
    """计算mAP@0.5"""
    aps = []

    for cls in range(n_classes):
        cls_preds = []
        cls_gts = []

        for img_idx, (preds, gts) in enumerate(zip(predictions, ground_truths)):
            img_pred = [p for p in preds if p['class'] == cls]
            img_gt = [g for g in gts if g['class'] == cls]

            cls_preds.append((img_idx, img_pred))
            cls_gts.append((img_idx, img_gt))

        # 收集所有预测
        all_preds = []
        for img_idx, preds in cls_preds:
            for pred in preds:
                all_preds.append({
                    'img_idx': img_idx,
                    'bbox': pred['bbox'],
                    'conf': pred['conf']
                })

        # 按置信度排序
        all_preds.sort(key=lambda x: x['conf'], reverse=True)

        # 统计TP/FP
        tp = np.zeros(len(all_preds))
        fp = np.zeros(len(all_preds))
        matched_gt = {}

        for pred_idx, pred in enumerate(all_preds):
            img_idx = pred['img_idx']
            bbox = pred['bbox']

            # 找对应的gt
            gts = [g for g in cls_gts[img_idx][1]]
            max_iou = 0
            max_gt_idx = -1

            for gt_idx, gt in enumerate(gts):
                if img_idx in matched_gt and gt_idx in matched_gt[img_idx]:
                    continue

                iou = box_iou(bbox, gt['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx

            if max_iou >= iou_threshold:
                tp[pred_idx] = 1
                if img_idx not in matched_gt:
                    matched_gt[img_idx] = set()
                matched_gt[img_idx].add(max_gt_idx)
            else:
                fp[pred_idx] = 1

        # 计算precision和recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        total_gt = sum(len(gts) for _, gts in cls_gts)
        if total_gt == 0:
            aps.append(0)
            continue

        recalls = tp_cumsum / total_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        ap = compute_ap(precisions, recalls)
        aps.append(ap)

    return np.mean(aps)


def box_iou(box1, box2):
    """计算IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter / (area1 + area2 - inter + 1e-6)
    return iou


# ============================================================
# 训练器
# ============================================================

class DetectionTrainer:
    def __init__(self, name, model, config):
        self.name = name
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.BCEWithLogitsLoss()

        self.history = {
            'epoch': [], 'loss': [], 'mAP@0.5': [], 'time': []
        }

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        n_batches = 0
        start_time = time.time()

        for images, boxes_list, labels_list, _ in dataloader:
            images = images.to(self.config.device)

            self.optimizer.zero_grad()

            # 前向
            outputs = self.model(images)

            # 简化损失：只训练分类部分
            B, C, H, W = outputs.shape
            targets = torch.zeros(B, 85, H, W).to(self.config.device)

            for b in range(B):
                boxes = boxes_list[b]
                labels = labels_list[b]
                for box, label in zip(boxes, labels):
                    if label < 80:
                        h_idx = min(int(box[1] * H), H-1)
                        w_idx = min(int(box[0] * W), W-1)
                        if h_idx < H and w_idx < W:
                            targets[b, 4, h_idx, w_idx] = 1
                            targets[b, 5+label, h_idx, w_idx] = 1

            loss = nn.functional.binary_cross_entropy_with_logits(
                outputs[:, :85], targets
            )

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        epoch_time = time.time() - start_time
        return total_loss / n_batches, epoch_time

    def evaluate(self, dataloader):
        self.model.eval()
        all_predictions = []
        all_ground_truths = []

        with torch.no_grad():
            for images, boxes_list, labels_list, _ in dataloader:
                images = images.to(self.config.device)
                outputs = self.model(images)

                B = outputs.shape[0]
                for b in range(B):
                    pred = outputs[b]  # [85, H, W]

                    boxes, classes, confs = parse_detection_output(
                        pred.cpu(), conf_thresh=self.config.conf_threshold
                    )

                    predictions = []
                    for box, cls, conf in zip(boxes, classes, confs):
                        predictions.append({
                            'bbox': box,
                            'class': cls,
                            'conf': conf
                        })
                    all_predictions.append(predictions)

                    ground_truths = []
                    boxes = boxes_list[b]
                    labels = labels_list[b]
                    for box, label in zip(boxes, labels):
                        ground_truths.append({
                            'bbox': box.tolist(),
                            'class': label.item()
                        })
                    all_ground_truths.append(ground_truths)

        # 计算mAP
        map_score = compute_map(all_predictions, all_ground_truths, self.config.iou_threshold, self.config.n_classes)
        return map_score

    def run(self, dataloader, epochs):
        print(f"\n{'='*60}")
        print(f"训练 {self.name}")
        print(f"{'='*60}")

        for epoch in range(1, epochs + 1):
            loss, epoch_time = self.train_epoch(dataloader)
            map_score = self.evaluate(dataloader)

            self.history['epoch'].append(epoch)
            self.history['loss'].append(loss)
            self.history['mAP@0.5'].append(map_score)
            self.history['time'].append(epoch_time)

            print(f"  Epoch {epoch:2d}/{epochs}: Loss={loss:.4f}, mAP@0.5={map_score:.4f}, Time={epoch_time:.2f}s")

        return self.history


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 70)
    print("Hebbian CNN 目标检测实验")
    print("对比: Standard CNN vs Hebbian CNN (COCO128)")
    print("=" * 70)

    config = Config()
    print(f"\n设备: {config.device}")
    print(f"图像大小: {config.img_size}x{config.img_size}")
    print(f"类别数: {config.n_classes}")

    # 加载数据集
    print("\n[1] 加载COCO128数据集...")

    dataset = COCO128DetectionDataset('./data/coco128', split='train', max_samples=128)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)

    print(f"  样本数: {len(dataset)}")

    # 统计标注
    total_boxes = 0
    for _, boxes, _, _ in dataset:
        total_boxes += len(boxes)
    print(f"  总标注框: {total_boxes}")

    # 创建模型
    print("\n[2] 创建模型...")

    std_model = DetectionCNN(n_classes=80)
    heb_model = HebbianDetectionCNN(n_classes=80)

    print(f"  Standard CNN: {sum(p.numel() for p in std_model.parameters()):,} 参数")
    print(f"  Hebbian CNN: {sum(p.numel() for p in heb_model.parameters()):,} 参数")

    # 训练
    print("\n[3] 开始训练...")

    trainer_std = DetectionTrainer('Standard CNN (检测)', std_model, config)
    history_std = trainer_std.run(dataloader, config.epochs)

    trainer_heb = DetectionTrainer('Hebbian CNN (检测)', heb_model, config)
    history_heb = trainer_heb.run(dataloader, config.epochs)

    # ============================================================
    # 结果汇总
    # ============================================================

    print("\n" + "=" * 70)
    print("实验结果汇总")
    print("=" * 70)

    print(f"\n{'模型':<25} {'最终mAP@0.5':<15} {'总时间(s)':<15}")
    print("-" * 55)
    print(f"{'Standard CNN':<25} {history_std['mAP@0.5'][-1]:.4f}{'':<10} {sum(history_std['time']):.2f}")
    print(f"{'Hebbian CNN':<25} {history_heb['mAP@0.5'][-1]:.4f}{'':<10} {sum(history_heb['time']):.2f}")

    # 计算提升
    map_std = history_std['mAP@0.5'][-1]
    map_heb = history_heb['mAP@0.5'][-1]
    improve = (map_heb - map_std) / map_std * 100 if map_std > 0 else 0

    print(f"\nmAP@0.5提升: {'+' if improve > 0 else ''}{improve:.2f}%")

    if map_heb > map_std:
        print("结论: Hebbian CNN 更优 ✅")
    elif map_heb < map_std:
        print("结论: Standard CNN 更优")
    else:
        print("结论: 两者相当")

    # ============================================================
    # 绘图
    # ============================================================

    print("\n[4] 生成图表...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss曲线
    ax = axes[0]
    ax.plot(history_std['epoch'], history_std['loss'], 'b-o', label='Standard CNN', linewidth=2)
    ax.plot(history_heb['epoch'], history_heb['loss'], 'r-s', label='Hebbian CNN', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # mAP曲线
    ax = axes[1]
    ax.plot(history_std['epoch'], history_std['mAP@0.5'], 'b-o', label='Standard CNN', linewidth=2)
    ax.plot(history_heb['epoch'], history_heb['mAP@0.5'], 'r-s', label='Hebbian CNN', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP@0.5')
    ax.set_title('Mean Average Precision @ IoU=0.5')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('detection_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  已保存: detection_comparison.png")

    # 保存Excel
    df = pd.DataFrame({
        'Model': ['Standard CNN', 'Hebbian CNN'],
        'Final Loss': [history_std['loss'][-1], history_heb['loss'][-1]],
        'Final mAP@0.5': [history_std['mAP@0.5'][-1], history_heb['mAP@0.5'][-1]],
        'Total Time (s)': [sum(history_std['time']), sum(history_heb['time'])],
    })
    df.to_excel('detection_results.xlsx', index=False)
    print("  已保存: detection_results.xlsx")

    print("\n" + "=" * 70)
    print("实验完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()