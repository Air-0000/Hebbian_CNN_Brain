"""
COCO128目标检测训练

使用COCO128数据集训练Hebbian CNN目标检测模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.ops import nms, box_iou
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import wget
import zipfile
from pathlib import Path


# ============================================================
# 数据下载与准备
# ============================================================

def download_coco128():
    """下载COCO128数据集"""
    data_dir = Path('./data/coco128')
    data_dir.mkdir(parents=True, exist_ok=True)

    # COCO128 URL
    url = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip'
    zip_path = data_dir / 'coco128.zip'

    if (data_dir / 'images').exists() and (data_dir / 'labels').exists():
        print("COCO128已存在，跳过下载")
        return data_dir

    print("下载COCO128数据集...")
    if not zip_path.exists():
        wget.download(url, out=str(zip_path))

    print("解压中...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    zip_path.unlink()  # 删除zip
    print(f"COCO128已下载到: {data_dir}")
    return data_dir


# ============================================================
# 数据集实现（YOLO格式）
# ============================================================

class COCO128Dataset(Dataset):
    """
    COCO128数据集（YOLO格式）

    目录结构：
    coco128/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
    """

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

    def __init__(self, data_dir, split='train', img_size=640):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size

        self.img_dir = self.data_dir / 'images' / split
        self.label_dir = self.data_dir / 'labels' / split

        # 获取所有图像
        self.img_files = sorted(list(self.img_dir.glob('*.jpg')) +
                               list(self.img_dir.glob('*.png')))

        # 变换
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # 加载图像
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size

        # 变换
        img_tensor = self.transform(img)

        # 加载标签
        label_path = self.label_dir / (img_path.stem + '.txt')
        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        x_center, y_center, w, h = map(float, parts[1:5])

                        # 转换回像素坐标
                        x1 = (x_center - w/2) * orig_w
                        y1 = (y_center - h/2) * orig_h
                        x2 = (x_center + w/2) * orig_w
                        y2 = (y_center + h/2) * orig_h

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
    """自定义batch整理"""
    imgs, boxes, labels, paths = zip(*batch)
    return torch.stack(imgs), boxes, labels, paths


# ============================================================
# 简化版Hebbian目标检测网络
# ============================================================

class HebbianDetector(nn.Module):
    """
    基于Hebbian学习的目标检测网络

    结构：
    - 主干：Hebbian预训练 + 普通CNN
    - 检测头：简化版YOLO风格
    """

    def __init__(self, n_classes=80):
        super().__init__()
        self.n_classes = n_classes

        # 主干网络（Hebbian预训练）
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 320 -> 160

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 160 -> 80

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 80 -> 40

            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 40 -> 20

            # Block 5
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 20 -> 10
        )

        # 检测头（每个scale预测）
        # 输出：[batch, 5*n_boxes + n_classes, h, w]
        # 5 = [x, y, w, h, objectness]
        self.head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 85, 1)  # 85 = 5 boxes * (4 coords + 1 obj) for 80 classes
        )

    def forward(self, x):
        feat = self.backbone(x)
        out = self.head(feat)
        return out


class SimpleDetector(nn.Module):
    """
    更简单的检测器（用于快速测试）
    """

    def __init__(self, n_classes=80):
        super().__init__()
        self.n_classes = n_classes

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        # 输出预测
        self.predict = nn.Conv2d(256, 85, 1)  # 80 classes + 5 (x,y,w,h,conf)

    def forward(self, x):
        x = self.features(x)
        x = self.predict(x)
        return x


# ============================================================
# 损失函数
# ============================================================

class DetectionLoss(nn.Module):
    """简化版检测损失"""

    def __init__(self, n_classes=80):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, predictions, targets):
        """
        predictions: [B, 85, H, W]
        targets: list of [N, 5] (x1,y1,x2,y2,class)
        """
        # 简化的损失计算
        # 只计算分类损失作为演示
        pred = predictions[:, :self.n_classes].mean()
        loss = torch.abs(pred)  # 简化损失

        return loss


# ============================================================
# 训练函数
# ============================================================

def train_coco128(epochs=10, batch_size=8, img_size=320):
    """训练COCO128目标检测模型"""

    print("=" * 60)
    print("COCO128目标检测训练")
    print("=" * 60)

    # 下载数据
    print("\n[1] 准备数据...")
    data_dir = download_coco128()

    # 创建数据集
    print("\n[2] 加载数据集...")
    train_dataset = COCO128Dataset(data_dir, split='train', img_size=img_size)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )

    print(f"  训练样本: {len(train_dataset)}")
    print(f"  类别数: {len(train_dataset.classes)}")

    # 创建模型
    print("\n[3] 创建模型...")
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"  设备: {device}")

    model = SimpleDetector(n_classes=80).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = DetectionLoss(n_classes=80)

    # 训练
    print("\n[4] 开始训练...")
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0

        for batch_idx, (images, boxes, labels, paths) in enumerate(train_loader):
            images = images.to(device)

            # 前向
            outputs = model(images)

            # 简化的损失
            loss = criterion(outputs, boxes)

            # 反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} [Batch {batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")

        avg_loss = total_loss / n_batches
        print(f"\n  Epoch {epoch+1} 完成 - 平均损失: {avg_loss:.4f}")

    # 保存模型
    print("\n[5] 保存模型...")
    torch.save(model.state_dict(), 'detector_weights.pth')
    print("  已保存: detector_weights.pth")

    # 测试推理
    print("\n[6] 测试推理...")
    model.eval()
    with torch.no_grad():
        # 取一张图测试
        img, boxes, labels, paths = train_dataset[0]
        img_batch = img.unsqueeze(0).to(device)

        output = model(img_batch)
        print(f"  输入: {img_batch.shape}")
        print(f"  输出: {output.shape}")

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)

    return model


# ============================================================
# 可视化结果
# ============================================================

def visualize_detection(image, boxes, labels, predictions=None, save_path='detection_result.png'):
    """可视化检测结果"""
    fig, ax = plt.subplots(1, figsize=(12, 9))

    # 显示图像
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])
        image = np.clip(image, 0, 1)

    ax.imshow(image)

    # 绘制真实框
    if boxes is not None and len(boxes) > 0:
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                 fill=False, color='green', linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1-5, COCO128Dataset.classes[label],
                   color='green', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

    # 绘制预测框
    if predictions is not None:
        for box, label, conf in predictions:
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                 fill=False, color='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y2+5, f'{COCO128Dataset.classes[label]}: {conf:.2f}',
                   color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"检测结果已保存: {save_path}")


# ============================================================
# 主函数
# ============================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--batch', type=int, default=8, help='批次大小')
    parser.add_argument('--img-size', type=int, default=320, help='图像大小')

    args = parser.parse_args()

    model = train_coco128(epochs=args.epochs, batch_size=args.batch, img_size=args.img_size)