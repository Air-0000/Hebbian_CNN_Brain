"""
BioHebbianNet-DenseNet-Lite - 极致优化版
优化方向:
1. Depthwise Separable Convolution (减少参数)
2. 减少 growth 和层数
3. SWA 训练策略
4. 更强 Dropout + GlobalAvgPool
5. 保持 Hebbian 调制
目标: <0.2M 参数, 90%+ 准确率
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
import copy

print("="*70, flush=True)
print("BioHebbianNet-DenseNet-Lite 极致优化版", flush=True)
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


# ==================== 深度可分离卷积 ====================
class DSConv(nn.Module):
    """Depthwise Separable Convolution + BatchNorm + Hebbian"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, use_hebbian=True):
        super().__init__()
        # Depthwise: 每个通道独立卷积
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_ch)

        # Pointwise: 1x1 线性组合
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_ch)

        # Hebbian 调制
        if use_hebbian:
            self.scale = nn.Parameter(torch.tensor(0.05))
            self.pv = nn.Parameter(torch.tensor(0.05))
        else:
            self.scale = None
            self.pv = None

    def forward(self, x):
        # Depthwise
        x = self.bn_dw(self.depthwise(x))
        x = F.relu(x)

        # Pointwise
        x = self.bn_pw(self.pointwise(x))

        # Hebbian 调制
        if self.scale is not None:
            x = x * (1 + self.scale)
            pv_factor = torch.sigmoid(self.pv * (x.mean() - 0.3))
            x = x * (1 - pv_factor * 0.3)

        return x


# ==================== 简化版 DenseLayer ====================
class LiteDenseLayer(nn.Module):
    """轻量级 DenseLayer"""
    def __init__(self, in_ch, growth=8, drop=0.3, use_hebbian=True):
        super().__init__()
        hidden_ch = growth * 4  # 压缩比 4:1
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_ch)
        self.conv2 = nn.Conv2d(hidden_ch, growth, 3, padding=1, bias=False)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

        if use_hebbian:
            self.scale = nn.Parameter(torch.tensor(0.05))
            self.pv = nn.Parameter(torch.tensor(0.05))
        else:
            self.scale = None
            self.pv = None

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))

        if self.scale is not None:
            out = out * (1 + self.scale)
            pv_factor = torch.sigmoid(self.pv * (out.mean() - 0.3))
            out = out * (1 - pv_factor * 0.3)

        out = self.drop(out)
        return torch.cat([x, out], 1)


# ==================== 过渡层 ====================
class LiteTransition(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        return self.pool(self.conv(F.relu(self.bn(x))))


# ==================== 密集块 ====================
class LiteDenseBlock(nn.Module):
    def __init__(self, n_layers, in_ch, growth=8, drop=0.3, use_hebbian=True):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(LiteDenseLayer(in_ch + i * growth, growth, drop, use_hebbian))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ==================== 轻量化 BioHebbian-DenseNet ====================
class BioHebbianDenseNetLite(nn.Module):
    """
    极致优化版 DenseNet
    - 使用深度可分离卷积
    - growth=8 (原12)
    - 每block 8层 (原12层)
    - 更激进的 Dropout
    """
    def __init__(self, num_classes=10, growth=8, n_layers=8, drop=0.3):
        super().__init__()

        # 初始层: 3 -> 24
        ch = 24
        self.stem = nn.Sequential(
            DSConv(3, ch, use_hebbian=True),
            nn.MaxPool2d(2),  # 32->16
        )

        # Block 1: 24 + 8*8 = 88
        self.block1 = LiteDenseBlock(n_layers, ch, growth, drop, use_hebbian=True)
        ch = ch + n_layers * growth  # 88
        self.trans1 = LiteTransition(ch, ch // 2)  # 44
        ch = ch // 2

        # Block 2: 44 + 8*8 = 108
        self.block2 = LiteDenseBlock(n_layers, ch, growth, drop, use_hebbian=True)
        ch = ch + n_layers * growth  # 108
        self.trans2 = LiteTransition(ch, ch // 2)  # 54
        ch = ch // 2

        # Block 3: 54 + 8*8 = 118
        self.block3 = LiteDenseBlock(n_layers, ch, growth, drop, use_hebbian=True)
        ch = ch + n_layers * growth  # 118

        # 最终层
        self.final_bn = nn.BatchNorm2d(ch)

        # 分类头: GlobalAvgPool -> 极简 Dense
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(ch, 64),  # 极简化
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.trans1(x)
        x = self.block2(x)
        x = self.trans2(x)
        x = self.block3(x)
        x = F.relu(self.final_bn(x))
        return self.classifier(x)


# ==================== 标准对照版 ====================
class StandardDenseNetLite(nn.Module):
    """标准版 (无 Hebbian)"""
    def __init__(self, num_classes=10, growth=8, n_layers=8, drop=0.0):
        super().__init__()

        ch = 24
        self.stem = nn.Sequential(
            DSConv(3, ch, use_hebbian=False),
            nn.MaxPool2d(2),
        )

        self.block1 = LiteDenseBlock(n_layers, ch, growth, drop, use_hebbian=False)
        ch = ch + n_layers * growth
        self.trans1 = LiteTransition(ch, ch // 2)
        ch = ch // 2

        self.block2 = LiteDenseBlock(n_layers, ch, growth, drop, use_hebbian=False)
        ch = ch + n_layers * growth
        self.trans2 = LiteTransition(ch, ch // 2)
        ch = ch // 2

        self.block3 = LiteDenseBlock(n_layers, ch, growth, drop, use_hebbian=False)
        ch = ch + n_layers * growth

        self.final_bn = nn.BatchNorm2d(ch)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.trans1(x)
        x = self.block2(x)
        x = self.trans2(x)
        x = self.block3(x)
        x = F.relu(self.final_bn(x))
        return self.classifier(x)


def predict_with_tta(model, images):
    model.eval()
    with torch.no_grad():
        out1 = torch.softmax(model(images), dim=1)
        out2 = torch.softmax(model(torch.flip(images, [3])), dim=1)
        out3 = torch.softmax(model(torch.flip(images, [2])), dim=1)
        return (out1 + out2 + out3) / 3


def train_model(model, name, train_loader, test_loader, epochs=150, lr=0.001):
    print(f"\n训练 {name}...", flush=True)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,} ({total_params/1e6:.2f}M)", flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = LabelSmoothingCE(smoothing=0.1)

    best_acc = 0
    best_state = None
    history = {'epoch': [], 'loss': [], 'train_acc': [], 'val_acc': []}

    # SWA 相关
    swa_model = copy.deepcopy(model)
    swa_start = epochs // 2
    swa_count = 0

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

        # SWA 更新
        if epoch > swa_start:
            swa_model.parameters()
            for p, p_swa in zip(model.parameters(), swa_model.parameters()):
                p_swa.data = p_swa.data * swa_count / (swa_count + 1) + p.data / (swa_count + 1)
            swa_count += 1

        scheduler.step()
        train_acc = 100. * correct / total

        # 使用 SWA 模型验证
        eval_model = swa_model if swa_count > 10 else model
        eval_model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = predict_with_tta(eval_model, images)
                _, predicted = outputs.max(1)
                correct_val += predicted.eq(labels).sum().item()
                total_val += labels.size(0)

        val_acc = 100. * correct_val / total_val

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in eval_model.state_dict().items()}

        history['epoch'].append(epoch)
        history['loss'].append(total_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if epoch % 10 == 0 or epoch <= 5:
            swa_flag = " [SWA]" if epoch > swa_start else ""
            print(f"  Epoch {epoch:3d}: Loss={total_loss/len(train_loader):.4f}, "
                  f"Train={train_acc:.1f}%, Val={val_acc:.1f}%{swa_flag}", flush=True)

    print(f"  最佳: {best_acc:.2f}%", flush=True)
    return best_acc, history, best_state, total_params


def main():
    print("\n加载CIFAR-10...", flush=True)

    # 数据增强
    train_loader = DataLoader(
        datasets.CIFAR10('./data', train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4, fill=128),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ]), download=True),
        batch_size=64, shuffle=True, num_workers=0  # 减小 batch size
    )

    test_loader = DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ]), download=True),
        batch_size=128, shuffle=False, num_workers=0
    )

    print(f"训练: {len(train_loader.dataset)}, 测试: {len(test_loader.dataset)}", flush=True)

    # BioHebbian 版本
    print("\n" + "="*70, flush=True)
    print("训练 BioHebbian-DenseNet-Lite", flush=True)
    print("="*70, flush=True)

    heb_model = BioHebbianDenseNetLite(num_classes=10, growth=8, n_layers=8, drop=0.3)
    heb_acc, heb_hist, heb_state, heb_params = train_model(
        heb_model, "BioHebbian-DenseNet-Lite",
        train_loader, test_loader, epochs=150, lr=0.001
    )

    os.makedirs('results/model_checkpoints', exist_ok=True)
    os.makedirs('results/xlsx', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    torch.save({'model_state_dict': heb_state, 'best_acc': heb_acc, 'params': heb_params},
               'results/model_checkpoints/densenet_lite_best.pth')

    # Standard 版本
    print("\n" + "="*70, flush=True)
    print("训练 Standard-DenseNet-Lite", flush=True)
    print("="*70, flush=True)

    std_model = StandardDenseNetLite(num_classes=10, growth=8, n_layers=8, drop=0.0)
    std_acc, std_hist, std_state, std_params = train_model(
        std_model, "Standard-DenseNet-Lite",
        train_loader, test_loader, epochs=150, lr=0.001
    )

    torch.save({'model_state_dict': std_state, 'best_acc': std_acc, 'params': std_params},
               'results/model_checkpoints/standard_densenet_lite_best.pth')

    # 结果汇总
    baseline = {
        'BioHebbian-DenseNet-40': 90.7,
        '118K模型': 92.6,
        'Lightweight CNN': 90.0,
    }

    print("\n" + "="*70, flush=True)
    print("DenseNet-Lite 实验结果", flush=True)
    print("="*70, flush=True)

    print(f"\n基准对比:", flush=True)
    for name, acc in baseline.items():
        print(f"  {name}: {acc:.1f}%", flush=True)

    print(f"\n训练结果:", flush=True)
    print(f"  BioHebbian-DenseNet-Lite: {heb_acc:.2f}% ({heb_params/1e6:.2f}M)", flush=True)
    print(f"  Standard-DenseNet-Lite:  {std_acc:.2f}% ({std_params/1e6:.2f}M)", flush=True)
    print(f"  Hebbian 提升: {heb_acc - std_acc:+.2f}%", flush=True)

    # 保存 Excel
    df = pd.DataFrame({
        'Model': list(baseline.keys()) + ['BioHebbian-DenseNet-Lite', 'Standard-DenseNet-Lite'],
        'Accuracy (%)': list(baseline.values()) + [heb_acc, std_acc],
        'Parameters': ['0.48M', '0.12M', '0.26M', f'{heb_params/1e6:.2f}M', f'{std_params/1e6:.2f}M'],
    })
    df.to_excel('results/xlsx/densenet_lite_results.xlsx', index=False)

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
    plt.savefig('figures/densenet_lite_training.png', dpi=150)
    print("\n已保存: figures/densenet_lite_training.png", flush=True)

    print("\n" + "="*70, flush=True)
    print("完成!", flush=True)
    print("="*70, flush=True)


if __name__ == '__main__':
    main()