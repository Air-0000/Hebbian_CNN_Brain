"""
BioHebbianNet v3 - 93%准确率版本
训练完成后保存 .pth 权重文件
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

print("="*70, flush=True)
print("BioHebbianNet v3 - 93%版本", flush=True)
print("="*70, flush=True)

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"设备: {device}", flush=True)


# ============================================================
# 数据增强
# ============================================================

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


# ============================================================
# 模型 - 93%版本 (每Block 1层卷积)
# ============================================================

class NeuroBlock(nn.Module):
    """带神经科学调制的卷积块"""
    def __init__(self, in_ch, out_ch, use_neuro=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if use_neuro:
            self.scale = nn.Parameter(torch.tensor(0.1))
            self.pv = nn.Parameter(torch.tensor(0.1))
        else:
            self.scale = None
            self.pv = None

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.scale is not None:
            x = x * (1 + self.scale)
            pv_factor = torch.sigmoid(self.pv * (x.mean() - 0.3))
            x = x * (1 - pv_factor * 0.3)

        return F.relu(x)


class BioHebbianNetV3(nn.Module):
    """93%版本: 每Block 1层卷积"""
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 32x32 → 16x16
            NeuroBlock(3, 64, use_neuro=True),
            nn.MaxPool2d(2),

            # Block 2: 16x16 → 8x8
            NeuroBlock(64, 128, use_neuro=True),
            nn.MaxPool2d(2),

            # Block 3: 8x8 → 4x4
            NeuroBlock(128, 256, use_neuro=True),
            nn.MaxPool2d(2),

            # Block 4: 4x4 → 2x2
            NeuroBlock(256, 512, use_neuro=False),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def predict_with_tta(model, images, device):
    model.eval()
    with torch.no_grad():
        out1 = torch.softmax(model(images), dim=1)
        out2 = torch.softmax(model(torch.flip(images, [3])), dim=1)
        return (out1 + out2) / 2


# ============================================================
# 训练函数 - 带保存模型
# ============================================================

def train_model(model, name, train_loader, test_loader, epochs=50, lr=0.001, use_cutmix=True):
    print(f"\n训练 {name}...", flush=True)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,} ({total_params/1e6:.2f}M)", flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = LabelSmoothingCE(smoothing=0.1)

    best_acc = 0
    best_model_state = None
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

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        history['epoch'].append(epoch)
        history['loss'].append(total_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}: Loss={total_loss/len(train_loader):.4f}, "
                  f"Train={train_acc:.1f}%, Val={val_acc:.1f}%", flush=True)

    print(f"  最佳: {best_acc:.2f}%", flush=True)
    return best_acc, history, best_model_state, total_params


# ============================================================
# 主函数
# ============================================================

def main():
    print("\n加载CIFAR-10...", flush=True)

    train_loader = DataLoader(
        datasets.CIFAR10('./data', train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4, fill=128),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ]), download=True),
        batch_size=128, shuffle=True, num_workers=0
    )

    test_loader = DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ]), download=True),
        batch_size=256, shuffle=False, num_workers=0
    )

    print(f"训练: {len(train_loader.dataset)}, 测试: {len(test_loader.dataset)}", flush=True)

    print("\n" + "="*70, flush=True)
    print("训练 BioHebbianNet v3 (50 epochs)", flush=True)
    print("="*70, flush=True)

    start = time.time()
    best_acc, history, best_state, params = train_model(
        BioHebbianNetV3(), "BioHebbianNet v3",
        train_loader, test_loader, epochs=50, lr=0.001
    )
    total_time = time.time() - start

    # 保存最佳模型
    os.makedirs('results/model_checkpoints', exist_ok=True)
    save_path = 'results/model_checkpoints/bio_hebbian_v3_best.pth'
    torch.save({
        'model_state_dict': best_state,
        'best_acc': best_acc,
        'params': params,
        'epochs': 50
    }, save_path)
    print(f"\n已保存: {save_path}", flush=True)

    # 结果
    print("\n" + "="*70, flush=True)
    print("结果", flush=True)
    print("="*70, flush=True)
    print(f"BioHebbianNet v3: {best_acc:.2f}%", flush=True)
    print(f"参数量: {params:,} ({params/1e6:.2f}M)", flush=True)
    print(f"训练时间: {total_time:.1f}s ({total_time/60:.1f}min)", flush=True)

    # 保存Excel
    os.makedirs('results/xlsx', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    df = pd.DataFrame([{
        'Model': 'BioHebbianNet v3',
        'Accuracy (%)': best_acc,
        'Parameters': f'{params/1e6:.2f}M'
    }])
    df.to_excel('results/xlsx/bio_hebbian_v3_results.xlsx', index=False)
    print("已保存: results/xlsx/bio_hebbian_v3_results.xlsx", flush=True)

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['epoch'], history['loss'], 'r-')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)

    ax2 = axes[0].twinx()
    ax2.plot(history['epoch'], history['val_acc'], 'b-', label='Val Acc')
    ax2.set_ylabel('Val Acc (%)', color='b')

    axes[1].plot(history['epoch'], history['train_acc'], 'r-', label='Train')
    axes[1].plot(history['epoch'], history['val_acc'], 'b-', label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/bio_hebbian_v3_training.png', dpi=150, bbox_inches='tight')
    print("已保存: figures/bio_hebbian_v3_training.png", flush=True)

    print("\n" + "="*70, flush=True)
    print("训练完成!", flush=True)
    print("="*70, flush=True)


if __name__ == '__main__':
    main()