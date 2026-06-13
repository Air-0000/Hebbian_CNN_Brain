"""
BioHebbianResNet - 简化版 (快速训练)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os

print("="*70)
print("BioHebbianResNet - ResNet架构 + Hebbian调制")
print("="*70)

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"设备: {device}")


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    x_mixed = x.clone()
    x_mixed[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return x_mixed, y, y[index], lam


class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_preds, dim=-1))


class HebbianResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_hebbian=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )

        if use_hebbian:
            self.scale = nn.Parameter(torch.tensor(0.1))
            self.pv = nn.Parameter(torch.tensor(0.1))
        else:
            self.scale = None
            self.pv = None

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)

        if self.scale is not None:
            out = out * (1 + self.scale)
            pv_factor = torch.sigmoid(self.pv * (out.mean() - 0.3))
            out = out * (1 - pv_factor * 0.3)

        return F.relu(out)


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class BioHebbianResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.stage1 = nn.Sequential(
            HebbianResBlock(64, 64, use_hebbian=True),
            HebbianResBlock(64, 64, use_hebbian=True),
            HebbianResBlock(64, 64, use_hebbian=True),
        )

        self.stage2 = nn.Sequential(
            HebbianResBlock(64, 128, stride=2, use_hebbian=True),
            HebbianResBlock(128, 128, use_hebbian=True),
            HebbianResBlock(128, 128, use_hebbian=True),
        )

        self.stage3 = nn.Sequential(
            HebbianResBlock(128, 256, stride=2, use_hebbian=True),
            HebbianResBlock(256, 256, use_hebbian=True),
            HebbianResBlock(256, 256, use_hebbian=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class StandardResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.stage1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64), BasicBlock(64, 64))
        self.stage2 = nn.Sequential(BasicBlock(64, 128, stride=2), BasicBlock(128, 128), BasicBlock(128, 128))
        self.stage3 = nn.Sequential(BasicBlock(128, 256, stride=2), BasicBlock(256, 256), BasicBlock(256, 256))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def predict_with_tta(model, images, device):
    model.eval()
    with torch.no_grad():
        out1 = torch.softmax(model(images), dim=1)
        out2 = torch.softmax(model(torch.flip(images, [3])), dim=1)
        return (out1 + out2) / 2


def train_model(model, name, train_loader, test_loader, epochs=50, lr=0.001, use_cutmix=True):
    print(f"\n训练 {name}...")
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = LabelSmoothingCE(smoothing=0.1)

    best_acc = 0
    history = {'epoch': [], 'loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            if use_cutmix and np.random.random() < 0.5:
                images, labels_a, labels_b, lam = cutmix_data(images, labels)
                outputs = model(images)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = 100. * correct / total

        model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = predict_with_tta(model, images, device)
                _, predicted = outputs.max(1)
                correct_val += predicted.eq(labels).sum().item()
                total_val += labels.size(0)

        val_acc = 100. * correct_val / total_val
        best_acc = max(best_acc, val_acc)

        history['epoch'].append(epoch)
        history['loss'].append(total_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"  Epoch {epoch:2d}: Loss={total_loss/len(train_loader):.4f}, Train={train_acc:.1f}%, Val={val_acc:.1f}%")

    print(f"  最佳: {best_acc:.2f}%")
    return best_acc, history


def main():
    print("\n加载CIFAR-10...")

    train_loader = DataLoader(
        datasets.CIFAR10('./data', train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4, fill=128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ]), download=False),
        batch_size=128, shuffle=True, num_workers=0
    )

    test_loader = DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ]), download=False),
        batch_size=256, shuffle=False, num_workers=0
    )

    print(f"训练: {len(train_loader.dataset)}, 测试: {len(test_loader.dataset)}")

    # 参数量
    bio_model = BioHebbianResNet()
    std_model = StandardResNet()
    p_bio = sum(p.numel() for p in bio_model.parameters())
    p_std = sum(p.numel() for p in std_model.parameters())

    print(f"\n参数量: BioHebbianResNet={p_bio:,}, StandardResNet={p_std:,}")

    # 训练
    print("\n" + "="*70)
    print("训练 Standard ResNet (50 epochs)...")
    print("="*70)
    start = time.time()
    std_acc, std_hist = train_model(StandardResNet(), "Standard ResNet", train_loader, test_loader, epochs=50)
    std_time = time.time() - start
    print(f"  时间: {std_time:.1f}s ({std_time/60:.1f}min)")

    print("\n" + "="*70)
    print("训练 BioHebbianResNet (50 epochs)...")
    print("="*70)
    start = time.time()
    bio_acc, bio_hist = train_model(BioHebbianResNet(), "BioHebbianResNet", train_loader, test_loader, epochs=50)
    bio_time = time.time() - start
    print(f"  时间: {bio_time:.1f}s ({bio_time/60:.1f}min)")

    # 结果
    print("\n" + "="*70)
    print("CIFAR-10 实验结果")
    print("="*70)
    print(f"Standard ResNet:   {std_acc:.2f}% ({p_std:,} 参数)")
    print(f"BioHebbianResNet: {bio_acc:.2f}% ({p_bio:,} 参数)")
    print(f"差异:             {bio_acc - std_acc:+.2f}%")

    # 保存
    os.makedirs('results/xlsx', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    df = pd.DataFrame({
        'Model': ['Standard ResNet', 'BioHebbianResNet'],
        'Accuracy (%)': [std_acc, bio_acc],
        'Parameters': [p_std, p_bio],
        'Train Time (s)': [std_time, bio_time]
    })
    df.to_excel('results/xlsx/bio_hebbian_resnet_results.xlsx', index=False)
    print("\n已保存: results/xlsx/bio_hebbian_resnet_results.xlsx")

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(std_hist['epoch'], std_hist['val_acc'], 'b-', label='Standard ResNet')
    axes[0].plot(bio_hist['epoch'], bio_hist['val_acc'], 'r-', label='BioHebbianResNet')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Validation Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(['Standard ResNet', 'BioHebbianResNet'], [std_acc, bio_acc], color=['steelblue', 'coral'])
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Final Comparison')
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate([std_acc, bio_acc]):
        axes[1].text(i, v + 0.5, f'{v:.2f}%', ha='center')

    plt.tight_layout()
    plt.savefig('figures/bio_hebbian_resnet_comparison.png', dpi=150, bbox_inches='tight')
    print("已保存: figures/bio_hebbian_resnet_comparison.png")

    print("\n" + "="*70)
    print("完成!")
    print("="*70)


if __name__ == '__main__':
    main()