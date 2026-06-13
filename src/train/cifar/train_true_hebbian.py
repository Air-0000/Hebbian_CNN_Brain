"""
真正的 Hebbian Learning 训练脚本
比较:
1. BioHebbianNet (Hebbian版) - 权重动态更新
2. StandardNet (标准版) - BP反向传播
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
import os

print("="*70, flush=True)
print("真正的 Hebbian Learning vs 标准训练", flush=True)
print("="*70, flush=True)

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"设备: {device}", flush=True)


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


# ==================== 从 true_hebbian_cnn 导入 ====================
from true_hebbian_cnn import (
    BioHebbianNetTrue, StandardNet,
    TrueHebbianConv2d, HebbianDenseLayer
)


def predict_with_tta(model, images):
    model.eval()
    with torch.no_grad():
        out1 = torch.softmax(model(images), dim=1)
        out2 = torch.softmax(model(torch.flip(images, [3])), dim=1)
        return (out1 + out2) / 2


def train_model(model, name, train_loader, test_loader, epochs=100, lr=0.001):
    print(f"\n训练 {name}...", flush=True)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    def count_hebbian_params(model):
        """统计Hebbian相关参数"""
        total = 0
        for name, param in model.named_parameters():
            if 'hebbian' in name or 'intra' in name:
                total += param.numel()
        return total

    hebbian_params = count_hebbian_params(model)

    print(f"  总参数量: {total_params:,} ({total_params/1e6:.2f}M)", flush=True)
    if hebbian_params > 0:
        print(f"  Hebbian参数: {hebbian_params:,} ({hebbian_params/total_params*100:.2f}%)", flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = LabelSmoothingCE(smoothing=0.1)

    best_acc = 0
    best_state = None
    history = {'epoch': [], 'loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            if np.random.random() < 0.5:
                images, labels_a, labels_b, lam = cutmix_data(images, labels)
                outputs = model(images)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
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
                outputs = predict_with_tta(model, images)
                _, predicted = outputs.max(1)
                correct_val += predicted.eq(labels).sum().item()
                total_val += labels.size(0)

        val_acc = 100. * correct_val / total_val

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        history['epoch'].append(epoch)
        history['loss'].append(total_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if epoch % 10 == 0 or epoch <= 5:
            print(f"  Epoch {epoch:3d}: Loss={total_loss/len(train_loader):.4f}, "
                  f"Train={train_acc:.1f}%, Val={val_acc:.1f}%", flush=True)

    print(f"  最佳: {best_acc:.2f}%", flush=True)
    return best_acc, history, best_state, total_params, hebbian_params


def main():
    print("\n加载CIFAR-10...", flush=True)

    train_loader = DataLoader(
        datasets.CIFAR10('./data', train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4, fill=128),
            transforms.RandomHorizontalFlip(),
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

    # Hebbian 版本
    print("\n" + "="*70, flush=True)
    print("训练 BioHebbianNet (真正Hebbian学习)", flush=True)
    print("="*70, flush=True)

    heb_model = BioHebbianNetTrue(num_classes=10, hebbian_lr=0.01)
    heb_acc, heb_hist, heb_state, heb_params, hebbian_params = train_model(
        heb_model, "BioHebbianNet",
        train_loader, test_loader, epochs=100, lr=0.001
    )

    # Standard 版本
    print("\n" + "="*70, flush=True)
    print("训练 StandardNet (标准BP)", flush=True)
    print("="*70, flush=True)

    std_model = StandardNet(num_classes=10)
    std_acc, std_hist, std_state, std_params, _ = train_model(
        std_model, "StandardNet",
        train_loader, test_loader, epochs=100, lr=0.001
    )

    # 保存结果
    os.makedirs('results/model_checkpoints', exist_ok=True)
    os.makedirs('results/xlsx', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    torch.save({'model_state_dict': heb_state, 'best_acc': heb_acc, 'params': heb_params},
               'results/model_checkpoints/true_hebbian_best.pth')
    torch.save({'model_state_dict': std_state, 'best_acc': std_acc, 'params': std_params},
               'results/model_checkpoints/standard_net_best.pth')

    # 结果汇总
    print("\n" + "="*70, flush=True)
    print("真正的 Hebbian Learning 实验结果", flush=True)
    print("="*70, flush=True)

    print(f"\n训练结果:", flush=True)
    print(f"  BioHebbianNet (Hebbian学习): {heb_acc:.2f}% ({heb_params/1e6:.2f}M, Hebbian参数{hebbian_params:,})", flush=True)
    print(f"  StandardNet (标准BP):       {std_acc:.2f}% ({std_params/1e6:.2f}M)", flush=True)
    print(f"  提升: {heb_acc - std_acc:+.2f}%", flush=True)

    # 保存 Excel
    df = pd.DataFrame({
        'Model': ['BioHebbianNet (Hebbian)', 'StandardNet (BP)'],
        'Accuracy (%)': [heb_acc, std_acc],
        'Total Parameters': [f'{heb_params/1e6:.2f}M', f'{std_params/1e6:.2f}M'],
        'Hebbian Parameters': [f'{hebbian_params:,}', '0'],
    })
    df.to_excel('results/xlsx/true_hebbian_results.xlsx', index=False)

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(heb_hist['epoch'], heb_hist['loss'], 'r-', label='Hebbian')
    axes[0].plot(std_hist['epoch'], std_hist['loss'], 'b-', label='Standard')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(heb_hist['epoch'], heb_hist['val_acc'], 'r-', label=f'Hebbian ({heb_acc:.1f}%)')
    axes[1].plot(std_hist['epoch'], std_hist['val_acc'], 'b-', label=f'Standard ({std_acc:.1f}%)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Val Accuracy (%)')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/true_hebbian_training.png', dpi=150)
    print("\n已保存: figures/true_hebbian_training.png", flush=True)

    print("\n" + "="*70, flush=True)
    print("完成!", flush=True)
    print("="*70, flush=True)


if __name__ == '__main__':
    main()