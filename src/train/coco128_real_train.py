"""
真实COCO128训练 - 解析YOLO格式标签

COCO128标签格式（YOLO）：
- 每行: <class> <x_center> <y_center> <width> <height>
- class是整数（0-79对应COCO的80类）
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


# ============================================================
# 配置
# ============================================================

class Config:
    epochs = 20
    batch_size = 16
    n_classes = 80
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'


# ============================================================
# COCO128数据集（解析YOLO标签）
# ============================================================

class COCO128Dataset(Dataset):
    """COCO128数据集 - 解析YOLO格式标签"""

    # COCO 80类（与YOLO索引对应）
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

    def __init__(self, data_dir, split='train', max_samples=500):
        self.data_dir = Path(data_dir)
        self.max_samples = max_samples

        # COCO128的split对应train2017/val2017
        split_dir = 'train2017' if split == 'train' else 'val2017'
        self.img_dir = self.data_dir / 'images' / split_dir
        self.label_dir = self.data_dir / 'labels' / split_dir

        # 获取所有图片
        self.img_files = sorted([
            f for f in self.img_dir.glob('*.jpg')
        ] + [f for f in self.img_dir.glob('*.png')]
        )[:max_samples]

        # 变换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 统计类别分布
        self.class_counts = np.zeros(80, dtype=int)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]

        # 加载图片
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)

        # 解析标签文件（YOLO格式）
        label_path = self.label_dir / (img_path.stem + '.txt')

        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()

            # 获取所有类别
            classes = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    classes.append(cls)
                    self.class_counts[cls] += 1

            # 如果有多个目标，取第一个类别（简化处理）
            if classes:
                label = classes[0]  # 第一个目标
            else:
                label = 0  # 默认第一类
        else:
            label = 0

        return img_tensor, label

    def get_class_distribution(self):
        """获取类别分布"""
        total = self.class_counts.sum()
        if total == 0:
            return {}
        return {self.classes[i]: self.class_counts[i] for i in range(80) if self.class_counts[i] > 0}


# ============================================================
# 模型
# ============================================================

class StandardCNN(nn.Module):
    """普通CNN"""
    def __init__(self, n_classes=80):
        super().__init__()
        self.features = nn.Sequential(
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
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class HebbianCNN(nn.Module):
    """Hebbian CNN"""
    def __init__(self, n_classes=80):
        super().__init__()

        # Hebbian参数
        self.hebbian_scale = nn.Parameter(torch.ones(4) * 0.02)
        self.hebbian_threshold = nn.Parameter(torch.zeros(4))

        self.features = nn.Sequential(
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
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x):
        # 分层特征提取 + Hebbian调制
        x1 = self.features[0:4](x)
        x2 = self.features[4:8](x1)
        x3 = self.features[8:12](x2)
        x4 = self.features[12:](x3)

        # Hebbian调制
        activity = [x1.mean(dim=(2,3)), x2.mean(dim=(2,3)), x3.mean(dim=(2,3)), x4.mean(dim=(2,3))]
        for i in range(4):
            mod = torch.sigmoid(activity[i].mean() - self.hebbian_threshold[i])
            modulation = 1 + self.hebbian_scale[i] * mod
            if i == 0:
                x1 = x1 * modulation
            elif i == 1:
                x2 = x2 * modulation
            elif i == 2:
                x3 = x3 * modulation
            else:
                x4 = x4 * modulation

        x = x4
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ============================================================
# 训练器
# ============================================================

class Trainer:
    def __init__(self, name, model, config):
        self.name = name
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

        self.history = {
            'epoch': [], 'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'time': []
        }

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()

        for images, labels in dataloader:
            images = images.to(self.config.device)
            labels = labels.to(self.config.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_time = time.time() - start_time
        return total_loss / len(dataloader), correct / total, epoch_time

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.config.device)
                labels = labels.to(self.config.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return total_loss / len(dataloader), correct / total

    def run(self, train_loader, val_loader, epochs):
        print(f"\n{'='*60}")
        print(f"训练 {self.name}")
        print(f"{'='*60}")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc, epoch_time = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['time'].append(epoch_time)

            print(f"  Epoch {epoch:2d}/{epochs}: Loss={train_loss:.4f}, "
                  f"Acc={train_acc*100:.1f}%, ValLoss={val_loss:.4f}, ValAcc={val_acc*100:.1f}%, "
                  f"Time={epoch_time:.2f}s")

        return self.history


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 70)
    print("真实COCO128训练 - Standard CNN vs Hebbian CNN")
    print("（解析YOLO格式标签）")
    print("=" * 70)

    config = Config()
    print(f"\n设备: {config.device}")

    # 加载数据集
    print("\n[1] 加载COCO128数据集...")
    data_dir = Path('./data/coco128')

    full_dataset = COCO128Dataset(data_dir, split='train', max_samples=500)

    # 划分训练/验证
    n_train = int(len(full_dataset) * 0.8)
    n_val = len(full_dataset) - n_train

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val]
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    print(f"  训练样本: {n_train}")
    print(f"  验证样本: {n_val}")
    print(f"  类别数: {config.n_classes}")

    # 打印类别分布
    print("\n  类别分布（Top 10）:")
    dist = full_dataset.get_class_distribution()
    sorted_dist = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:10]
    for cls, count in sorted_dist:
        print(f"    {cls}: {count}")

    # 创建模型
    print("\n[2] 创建模型...")

    std_model = StandardCNN(n_classes=80)
    heb_model = HebbianCNN(n_classes=80)

    print(f"  Standard CNN: {sum(p.numel() for p in std_model.parameters()):,} 参数")
    print(f"  Hebbian CNN: {sum(p.numel() for p in heb_model.parameters()):,} 参数")

    # 训练
    print("\n[3] 开始训练...")

    trainer_std = Trainer('Standard CNN', std_model, config)
    history_std = trainer_std.run(train_loader, val_loader, config.epochs)

    trainer_heb = Trainer('Hebbian CNN', heb_model, config)
    history_heb = trainer_heb.run(train_loader, val_loader, config.epochs)

    # ============================================================
    # 结果汇总
    # ============================================================

    print("\n" + "=" * 70)
    print("实验结果汇总")
    print("=" * 70)

    print(f"\n{'指标':<25} {'Standard CNN':<20} {'Hebbian CNN':<20}")
    print("-" * 65)
    print(f"{'Final Train Loss':<25} {history_std['train_loss'][-1]:<20.4f} {history_heb['train_loss'][-1]:<20.4f}")
    print(f"{'Final Train Acc':<25} {history_std['train_acc'][-1]*100:<20.1f}% {history_heb['train_acc'][-1]*100:<20.1f}%")
    print(f"{'Final Val Loss':<25} {history_std['val_loss'][-1]:<20.4f} {history_heb['val_loss'][-1]:<20.4f}")
    print(f"{'Final Val Acc':<25} {history_std['val_acc'][-1]*100:<20.1f}% {history_heb['val_acc'][-1]*100:<20.1f}%")
    print(f"{'Total Time (s)':<25} {sum(history_std['time']):<20.2f} {sum(history_heb['time']):<20.2f}")
    print("-" * 65)

    # 计算提升
    acc_std = history_std['val_acc'][-1]
    acc_heb = history_heb['val_acc'][-1]
    improve = (acc_heb - acc_std) / acc_std * 100 if acc_std > 0 else 0

    print(f"\n验证准确率提升: {'+' if improve > 0 else ''}{improve:.2f}%")

    if acc_heb > acc_std:
        print("结论: Hebbian CNN 更优 ✅")
    else:
        print("结论: Standard CNN 更优")

    # ============================================================
    # 绘图
    # ============================================================

    print("\n[4] 生成图表...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    ax = axes[0]
    ax.plot(history_std['epoch'], history_std['train_loss'], 'b-o', label='Std Train', markersize=4)
    ax.plot(history_std['epoch'], history_std['val_loss'], 'b--s', label='Std Val', markersize=4)
    ax.plot(history_heb['epoch'], history_heb['train_loss'], 'r-o', label='Heb Train', markersize=4)
    ax.plot(history_heb['epoch'], history_heb['val_loss'], 'r--s', label='Heb Val', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 准确率曲线
    ax = axes[1]
    ax.plot(history_std['epoch'], [a*100 for a in history_std['train_acc']], 'b-o', label='Std Train', markersize=4)
    ax.plot(history_std['epoch'], [a*100 for a in history_std['val_acc']], 'b--s', label='Std Val', markersize=4)
    ax.plot(history_heb['epoch'], [a*100 for a in history_heb['train_acc']], 'r-o', label='Heb Train', markersize=4)
    ax.plot(history_heb['epoch'], [a*100 for a in history_heb['val_acc']], 'r--s', label='Heb Val', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Training & Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('coco128_real_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  已保存: coco128_real_curves.png")

    # Excel
    results = {
        'Model': ['Standard CNN', 'Hebbian CNN'],
        'Final Train Loss': [history_std['train_loss'][-1], history_heb['train_loss'][-1]],
        'Final Train Acc (%)': [history_std['train_acc'][-1]*100, history_heb['train_acc'][-1]*100],
        'Final Val Loss': [history_std['val_loss'][-1], history_heb['val_loss'][-1]],
        'Final Val Acc (%)': [history_std['val_acc'][-1]*100, history_heb['val_acc'][-1]*100],
        'Total Time (s)': [sum(history_std['time']), sum(history_heb['time'])],
    }
    df = pd.DataFrame(results)
    df.to_excel('coco128_real_results.xlsx', index=False)
    print("  已保存: coco128_real_results.xlsx")

    print("\n" + "=" * 70)
    print("实验完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()