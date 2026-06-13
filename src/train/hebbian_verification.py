"""
Hebbian CNN 核心验证实验

问题：COCO128不适合做80类分类（每图多标签，高度不平衡）
解决：使用合成图像分类任务，正确验证Hebbian机制效果

实验设计：
1. 合成数据集：10类简单几何形状
2. 对比：Standard CNN vs Hebbian CNN vs Full Hebbian
3. 控制变量：相同架构，仅Hebbian调制不同
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
import random
from PIL import Image, ImageDraw


# ============================================================
# 配置
# ============================================================

class Config:
    epochs = 50
    batch_size = 16
    n_classes = 10
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    img_size = 64
    samples_per_class = 200  # 每类200样本
    seed = 42


# ============================================================
# 合成数据集生成器
# ============================================================

def generate_synthetic_dataset(config):
    """
    生成10类合成图像数据集
    类别：不同几何形状（圆形、三角形、方形等）+ 不同颜色 + 不同大小
    """

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    shapes = ['circle', 'square', 'triangle', 'rectangle', 'ellipse',
              'diamond', 'pentagon', 'hexagon', 'star', 'cross']
    colors = ['red', 'blue', 'green', 'yellow', 'orange',
              'purple', 'cyan', 'brown', 'pink', 'gray']

    color_rgb = {
        'red': (255, 0, 0), 'blue': (0, 0, 255), 'green': (0, 255, 0),
        'yellow': (255, 255, 0), 'orange': (255, 165, 0), 'purple': (128, 0, 128),
        'cyan': (0, 255, 255), 'brown': (139, 69, 19), 'pink': (255, 192, 203),
        'gray': (128, 128, 128)
    }

    all_images = []
    all_labels = []

    print("生成合成数据集...")
    for class_id, shape in enumerate(shapes):
        for i in range(config.samples_per_class):
            # 创建图像 (RGB格式，HWC)
            img = Image.new('RGB', (config.img_size, config.img_size), 'white')
            draw = ImageDraw.Draw(img)

            # 随机参数
            color = colors[random.randint(0, len(colors)-1)]
            rgb = color_rgb[color]

            # 位置和大小
            cx = random.randint(20, config.img_size - 20)
            cy = random.randint(20, config.img_size - 20)
            size = random.randint(15, 25)

            # 绘制形状
            if shape == 'circle':
                draw.ellipse([cx-size, cy-size, cx+size, cy+size], fill=rgb)
            elif shape == 'square':
                draw.rectangle([cx-size, cy-size, cx+size, cy+size], fill=rgb)
            elif shape == 'triangle':
                points = [(cx, cy-size), (cx-size, cy+size), (cx+size, cy+size)]
                draw.polygon(points, fill=rgb)
            elif shape == 'rectangle':
                draw.rectangle([cx-size, cy-int(size/2), cx+size, cy+int(size/2)], fill=rgb)
            elif shape == 'ellipse':
                draw.ellipse([cx-size*2, cy-size, cx+size*2, cy+size], fill=rgb)
            elif shape == 'diamond':
                points = [(cx, cy-size), (cx-size, cy), (cx, cy+size), (cx+size, cy)]
                draw.polygon(points, fill=rgb)
            elif shape == 'pentagon':
                points = [(cx, cy-size)] + [
                    (cx + size * np.cos(2*np.pi*i/5 - np.pi/2),
                     cy + size * np.sin(2*np.pi*i/5 - np.pi/2))
                    for i in range(5)
                ]
                draw.polygon(points, fill=rgb)
            elif shape == 'hexagon':
                points = [
                    (cx + size * np.cos(2*np.pi*i/6),
                     cy + size * np.sin(2*np.pi*i/6))
                    for i in range(6)
                ]
                draw.polygon(points, fill=rgb)
            elif shape == 'star':
                points = []
                for i in range(10):
                    r = size if i % 2 == 0 else size//2
                    angle_rad = 2*np.pi*i/10 - np.pi/2
                    points.append((cx + r*np.cos(angle_rad), cy + r*np.sin(angle_rad)))
                draw.polygon(points, fill=rgb)
            elif shape == 'cross':
                draw.rectangle([cx-size, cy-size//2, cx+size, cy+size//2], fill=rgb)
                draw.rectangle([cx-size//2, cy-size, cx+size//2, cy+size], fill=rgb)

            # 添加噪声
            pixels = np.array(img)
            noise = np.random.randint(-20, 20, pixels.shape, dtype=np.int16)
            pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(pixels)

            # 强制转换为tensor (C, H, W格式)
            img_tensor = transforms.ToTensor()(img)  # 这会返回(C, H, W)
            all_images.append(img_tensor)
            all_labels.append(class_id)

    # 转换为tensor
    X = torch.stack(all_images)  # 应该是(N, 3, H, W)
    y = torch.tensor(all_labels, dtype=torch.long)

    # 验证格式
    print(f"  X shape: {X.shape}")  # 应该是(N, 3, H, W)
    assert X.shape[1] == 3, f"Expected 3 channels, got {X.shape[1]}"

    # 打乱
    indices = torch.randperm(len(X))
    X = X[indices]
    y = y[indices]

    print(f"  生成了 {len(X)} 张图像，10类")
    print(f"  每类样本数: {config.samples_per_class}")
    print(f"  图像格式: (N, C, H, W) = {X.shape}")

    return X, y, shapes


class SyntheticDataset(Dataset):
    """合成数据集包装器"""

    def __init__(self, X, y, augment=False, normalize=False):
        self.X = X
        self.y = y
        self.augment = augment
        self.normalize = normalize
        if normalize:
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx].clone()  # 确保格式一致 (C, H, W)
        label = self.y[idx]

        # 数据增强
        if self.augment:
            # 随机翻转
            if random.random() > 0.5:
                img = torch.flip(img, [2])  # 翻转W
            # 随机旋转 (90度倍数)
            if random.random() > 0.5:
                k = random.randint(1, 3)
                img = torch.rot90(img, k, dims=[1, 2])  # 旋转H和W

        # 归一化
        if self.normalize:
            img = (img - self.mean) / self.std

        return img, label


# ============================================================
# 模型定义
# ============================================================

class BaseCNN(nn.Module):
    """基础CNN架构"""

    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 64 -> 32
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Block 2: 32 -> 16
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Block 3: 16 -> 8
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Block 4: 8 -> 4
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
    """Hebbian调制CNN"""

    def __init__(self, n_classes=10):
        super().__init__()
        # 可学习的Hebbian参数
        self.hebb_scale = nn.Parameter(torch.ones(4) * 0.05)
        self.hebb_threshold = nn.Parameter(torch.zeros(4))

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
        # 提取分层特征
        x1 = self.features[0:3](x)
        x2 = self.features[3:6](x1)
        x3 = self.features[6:9](x2)
        x4 = self.features[9:](x3)

        # 计算各层激活
        activity = [x1.mean(), x2.mean(), x3.mean(), x4.mean()]

        # Hebbian调制
        for i in range(4):
            # 调制强度基于激活水平
            activation = activity[i].item()
            threshold = self.hebb_threshold[i].item()

            # sigmoid调制
            modulation = torch.sigmoid(torch.tensor(activation - threshold))
            scale = self.hebb_scale[i]

            mod = 1 + scale * (modulation - 0.5) * 2

            if i == 0:
                x1 = x1 * mod
            elif i == 1:
                x2 = x2 * mod
            elif i == 2:
                x3 = x3 * mod
            else:
                x4 = x4 * mod

        x = x4
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class FullHebbianCNN(nn.Module):
    """完整Hebbian CNN（多层调制 + 竞争 + LTP/LTD）"""

    def __init__(self, n_classes=10):
        super().__init__()
        # Hebbian参数
        self.hebb_scale = nn.Parameter(torch.ones(4) * 0.06)
        self.hebb_threshold = nn.Parameter(torch.zeros(4))
        # LTP/LTD阈值
        self.ltp_threshold = nn.Parameter(torch.tensor([0.6, 0.5, 0.4, 0.3]))
        self.ltd_threshold = nn.Parameter(torch.tensor([0.3, 0.25, 0.2, 0.15]))
        # 竞争权重
        self.wta_ratio = 0.2

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
        x1 = self.features[0:3](x)
        x2 = self.features[3:6](x1)
        x3 = self.features[6:9](x2)
        x4 = self.features[9:](x3)

        layers = [x1, x2, x3, x4]
        activity = [l.mean() for l in layers]

        for i in range(4):
            activation = activity[i].item()
            scale = self.hebb_scale[i]
            ltp_t = self.ltp_threshold[i].item()
            ltd_t = self.ltd_threshold[i].item()

            # Hebbian调制
            hebb_mod = 1 + scale * (torch.sigmoid(torch.tensor(activation - self.hebb_threshold[i].item())) - 0.5) * 2

            # LTP/LTD调制
            if activation > ltp_t:
                ltp_mod = 1.05
            elif activation < ltd_t:
                ltp_mod = 0.95
            else:
                ltp_mod = 1.0

            # 综合调制
            mod = hebb_mod * ltp_mod

            layers[i] = layers[i] * mod

        x = layers[3]
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
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.epochs)
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        self.scheduler.step()
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

        best_val_acc = 0
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc, epoch_time = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['time'].append(epoch_time)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch

            print(f"  Epoch {epoch:2d}/{epochs}: Loss={train_loss:.4f}, "
                  f"Acc={train_acc*100:.1f}%, ValLoss={val_loss:.4f}, ValAcc={val_acc*100:.1f}%, "
                  f"Time={epoch_time:.2f}s")

        print(f"  最佳验证准确率: {best_val_acc*100:.1f}% (Epoch {best_epoch})")
        return self.history


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 70)
    print("Hebbian CNN 核心验证实验")
    print("使用合成数据集：10类几何形状，正确验证Hebbian机制")
    print("=" * 70)

    config = Config()
    print(f"\n设备: {config.device}")
    print(f"训练轮数: {config.epochs}")

    # 生成合成数据集
    print("\n[1] 生成合成数据集...")
    X, y, class_names = generate_synthetic_dataset(config)

    # 划分训练/验证/测试
    n_total = len(X)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)
    n_test = n_total - n_train - n_val

    train_dataset = SyntheticDataset(X[:n_train], y[:n_train], augment=True, normalize=True)
    val_dataset = SyntheticDataset(X[n_train:n_train+n_val], y[n_train:n_train+n_val], augment=False, normalize=True)
    test_dataset = SyntheticDataset(X[n_train+n_val:], y[n_train+n_val:], augment=False, normalize=True)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    print(f"\n  训练样本: {n_train}")
    print(f"  验证样本: {n_val}")
    print(f"  测试样本: {n_test}")
    print(f"  类别数: {config.n_classes}")

    # 显示类别分布
    print("\n  类别分布:")
    for i, name in enumerate(class_names):
        count = (y[:n_train] == i).sum().item()
        print(f"    {i}. {name}: {count}")

    # 创建模型
    print("\n[2] 创建模型...")

    models = {
        'Baseline (Standard CNN)': BaseCNN(n_classes=10),
        'Hebbian CNN': HebbianCNN(n_classes=10),
        'Full Hebbian CNN': FullHebbianCNN(n_classes=10),
    }

    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {params:,} 参数")

    # 训练所有模型
    print("\n[3] 开始训练...")

    results = {}
    for name, model in models.items():
        trainer = Trainer(name, model, config)
        history = trainer.run(train_loader, val_loader, config.epochs)
        results[name] = history

    # 测试集评估
    print("\n[4] 测试集评估...")

    test_results = {}
    for name, model in models.items():
        trainer = Trainer(name, model, config)
        _, test_acc = trainer.evaluate(test_loader)
        test_results[name] = test_acc
        print(f"  {name}: {test_acc*100:.2f}%")

    # ============================================================
    # 结果汇总
    # ============================================================

    print("\n" + "=" * 70)
    print("实验结果汇总")
    print("=" * 70)

    print(f"\n{'模型':<25} {'训练准确率':<15} {'验证准确率':<15} {'测试准确率':<15}")
    print("-" * 70)
    for name, history in results.items():
        train_acc = history['train_acc'][-1] * 100
        val_acc = history['val_acc'][-1] * 100
        test_acc = test_results[name] * 100
        print(f"{name:<25} {train_acc:.1f}%{'':<10} {val_acc:.1f}%{'':<10} {test_acc:.1f}%")

    # 计算提升
    baseline_val = results['Baseline (Standard CNN)']['val_acc'][-1] * 100
    baseline_test = test_results['Baseline (Standard CNN)'] * 100

    print(f"\n相对Baseline的提升:")
    print("-" * 50)
    for name in results.keys():
        val_improve = results[name]['val_acc'][-1] * 100 - baseline_val
        test_improve = test_results[name] * 100 - baseline_test
        sign = '+' if val_improve > 0 else ''
        print(f"  {name:<25}: Val {sign}{val_improve:.2f}%, Test {sign}{test_improve:.2f}%")

    # ============================================================
    # 绘图
    # ============================================================

    print("\n[5] 生成图表...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    colors = {'Baseline (Standard CNN)': '#3498DB', 'Hebbian CNN': '#E74C3C', 'Full Hebbian CNN': '#2ECC71'}

    # 训练损失
    ax = axes[0, 0]
    for name, history in results.items():
        ax.plot(history['epoch'], history['train_loss'], color=colors[name], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 验证损失
    ax = axes[0, 1]
    for name, history in results.items():
        ax.plot(history['epoch'], history['val_loss'], color=colors[name], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 训练准确率
    ax = axes[1, 0]
    for name, history in results.items():
        ax.plot(history['epoch'], [a*100 for a in history['train_acc']], color=colors[name], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Training Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 验证准确率
    ax = axes[1, 1]
    for name, history in results.items():
        ax.plot(history['epoch'], [a*100 for a in history['val_acc']], color=colors[name], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('hebbian_verification.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  已保存: hebbian_verification.png")

    # 准确率对比条形图
    fig, ax = plt.subplots(figsize=(8, 5))

    names = list(results.keys())
    val_accs = [results[n]['val_acc'][-1] * 100 for n in names]
    test_accs = [test_results[n] * 100 for n in names]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, val_accs, width, label='Validation', color='steelblue')
    bars2 = ax.bar(x + width/2, test_accs, width, label='Test', color='coral')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(' ', '\n') for n in names])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 标注数值
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('hebbian_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  已保存: hebbian_comparison.png")

    # 保存Excel
    df = pd.DataFrame({
        'Model': names,
        'Final Train Acc (%)': [results[n]['train_acc'][-1] * 100 for n in names],
        'Final Val Acc (%)': [results[n]['val_acc'][-1] * 100 for n in names],
        'Final Val Loss': [results[n]['val_loss'][-1] for n in names],
        'Test Acc (%)': [test_results[n] * 100 for n in names],
        'Total Time (s)': [sum(results[n]['time']) for n in names],
    })
    df.to_excel('hebbian_verification.xlsx', index=False)
    print("  已保存: hebbian_verification.xlsx")

    # ============================================================
    # 结论
    # ============================================================

    print("\n" + "=" * 70)
    print("实验结论")
    print("=" * 70)

    best_name = max(results.keys(), key=lambda k: results[k]['val_acc'][-1])
    best_val = results[best_name]['val_acc'][-1] * 100
    best_test = test_results[best_name] * 100

    print(f"\n最佳模型: {best_name}")
    print(f"验证准确率: {best_val:.2f}%")
    print(f"测试准确率: {best_test:.2f}%")
    print(f"相对Baseline提升: +{best_val - baseline_val:.2f}%")

    # Hebbian机制分析
    print("\nHebbian机制效果分析:")
    print("-" * 50)

    hebb_val = results['Hebbian CNN']['val_acc'][-1] * 100
    full_val = results['Full Hebbian CNN']['val_acc'][-1] * 100

    print(f"  简单Hebbian vs Baseline: {hebb_val - baseline_val:+.2f}%")
    print(f"  完整Hebbian vs Baseline: {full_val - baseline_val:+.2f}%")
    print(f"  完整Hebbian vs 简单Hebbian: {full_val - hebb_val:+.2f}%")

    print("\n" + "=" * 70)
    print("实验完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()