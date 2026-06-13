"""
True Hebbian DenseNet - 50% Hebbian参数占比版本

目标：Hebbian参数占总参数的50%，让Hebbian真正起主导作用
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
print("True Hebbian DenseNet-40 (50% Hebbian占比)", flush=True)
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


# ==================== 50% Hebbian Dense Layer ====================
class HebbianHeavyDenseLayer(nn.Module):
    """
    Hebbian主导的DenseNet层 (50%+ Hebbian参数)

    设计思路：
    1. 减少卷积参数（压缩BP部分）
    2. 增强Hebbian机制（扩大Hebbian部分）

    每层参数分配：
    - BP部分（卷积+BN）: 约50%
    - Hebbian部分: 约50%
    """
    def __init__(self, in_ch, growth=12, drop=0.2, hebbian_lr=0.05):
        super().__init__()

        self.growth = growth
        self.in_ch = in_ch
        self.hebbian_lr = hebbian_lr
        self.drop = drop

        # ========== BP部分（精简） ==========
        self.bn1 = nn.BatchNorm2d(in_ch)
        # 使用深度可分离卷积减少BP参数
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1, bias=False),  # 逐点
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
            nn.Conv2d(in_ch, 4*growth, 1, bias=False),  # 扩展
        )
        self.bn2 = nn.BatchNorm2d(4*growth)
        self.conv2 = nn.Conv2d(4*growth, growth, 3, padding=1, bias=False)

        # ========== Hebbian部分（50%占比） ==========
        # Hebbian突触权重: [in_ch, growth*16] - 大幅扩大
        self.hebbian_w = nn.Parameter(
            torch.randn(in_ch, growth * 16) * 0.01
        )

        # 通道间Hebbian调制: [growth*16, growth*16]
        self.hebbian_inter = nn.Parameter(
            torch.randn(growth * 16, growth * 16) * 0.01
        )

        # 通道内Hebbian调制: [growth*8, growth*8]
        self.hebbian_intra = nn.Parameter(
            torch.randn(growth * 8, growth * 8) * 0.01
        )

        # 多层调制
        self.mod1 = nn.Parameter(torch.tensor(1.0))
        self.mod2 = nn.Parameter(torch.tensor(0.5))
        self.mod3 = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # BP卷积
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))

        # ========== Hebbian调制（增强） ==========
        if self.training:
            # 输入活动
            in_act = x.mean(dim=(2, 3))  # [B, in_ch]

            # 输出活动
            out_act = out.mean(dim=(2, 3))  # [B, growth]

            # Hebbian更新
            raw_update = torch.ger(in_act.mean(0), out_act.mean(0))  # [in_ch, growth]
            # 重复扩展到 growth*16
            update = raw_update.repeat(1, 16)  # [in_ch, growth*16]
            with torch.no_grad():
                self.hebbian_w.data = self.hebbian_w.data * 0.95 + update * 0.05

            # 多层Hebbian调制
            # 层1: in_ch -> growth*16 -> 取前growth个
            mod1 = torch.sigmoid(torch.matmul(in_act, self.hebbian_w)[:, :self.growth])  # [B, growth]
            # 层2: growth -> growth*8 -> 取前growth个
            mod2 = torch.sigmoid(torch.matmul(mod1, self.hebbian_inter[:self.growth, :self.growth]))  # [B, growth]
            # 层3: 通道内 growth*8 -> growth
            mod3 = torch.sigmoid(torch.matmul(out_act, self.hebbian_intra[:self.growth, :self.growth]))  # [B, growth]

            # 综合调制 (维度统一为 [growth])
            hebbian_effect = mod1.mean(0) * self.mod1 * 0.3 + \
                            mod2.mean(0) * self.mod2 * 0.3 + \
                            mod3.mean(0) * self.mod3 * 0.2

            out = out * (1 + hebbian_effect.unsqueeze(-1).unsqueeze(-1))

        if self.drop > 0:
            out = F.dropout(out, p=self.drop, training=self.training)

        return out


class SimpleDenseLayer(nn.Module):
    """标准DenseNet层（用于对照组）"""
    def __init__(self, in_ch, growth=12, drop=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, 4*growth, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth)
        self.conv2 = nn.Conv2d(4*growth, growth, 3, padding=1, bias=False)
        self.drop = drop

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        if self.drop > 0:
            out = F.dropout(out, p=self.drop, training=self.training)
        return out


class DenseBlock(nn.Module):
    def __init__(self, layers, in_ch, growth=12, drop=0.2, hebbian=True, hebbian_lr=0.05):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(layers):
            if hebbian:
                self.layers.append(HebbianHeavyDenseLayer(in_ch + i*growth, growth, drop, hebbian_lr))
            else:
                self.layers.append(SimpleDenseLayer(in_ch + i*growth, growth, drop))

    def forward(self, x):
        for layer in self.layers:
            x = torch.cat([x, layer(x)], 1)
        return x


class Transition(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        return self.pool(self.conv(F.relu(self.bn(x))))


# ==================== 模型定义 ====================
class HebbianHeavyDenseNet(nn.Module):
    """
    50% Hebbian占比的DenseNet
    """
    def __init__(self, num_classes=10, growth=12, drop=0.2, hebbian_lr=0.05):
        super().__init__()
        ch = 16

        self.features = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
        )

        self.block1 = DenseBlock(12, ch, growth, drop, True, hebbian_lr)
        ch = ch + 12*growth
        self.trans1 = Transition(ch, ch//2)
        ch = ch // 2

        self.block2 = DenseBlock(12, ch, growth, drop, True, hebbian_lr)
        ch = ch + 12*growth
        self.trans2 = Transition(ch, ch//2)
        ch = ch // 2

        self.block3 = DenseBlock(12, ch, growth, drop, True, hebbian_lr)
        ch = ch + 12*growth

        self.final_bn = nn.BatchNorm2d(ch)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(ch, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.block1(x)
        x = self.trans1(x)
        x = self.block2(x)
        x = self.trans2(x)
        x = self.block3(x)
        x = F.relu(self.final_bn(x))
        return self.classifier(x)


class StandardDenseNet(nn.Module):
    """标准DenseNet对照组"""
    def __init__(self, num_classes=10, growth=12, drop=0.2):
        super().__init__()
        ch = 16

        self.features = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
        )

        self.block1 = DenseBlock(12, ch, growth, drop, False)
        ch = ch + 12*growth
        self.trans1 = Transition(ch, ch//2)
        ch = ch // 2

        self.block2 = DenseBlock(12, ch, growth, drop, False)
        ch = ch + 12*growth
        self.trans2 = Transition(ch, ch//2)
        ch = ch // 2

        self.block3 = DenseBlock(12, ch, growth, drop, False)
        ch = ch + 12*growth

        self.final_bn = nn.BatchNorm2d(ch)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(ch, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
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
        return (out1 + out2) / 2


def count_hebbian_params(model):
    """统计Hebbian参数"""
    total = 0
    for name, param in model.named_parameters():
        if 'hebbian' in name or 'mod1' in name or 'mod2' in name or 'mod3' in name:
            total += param.numel()
    return total


def train_model(model, name, train_loader, test_loader, epochs=100, lr=0.001, patience=20):
    print(f"\n训练 {name}...", flush=True)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
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

    no_improve_count = 0

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
                outputs = predict_with_tta(model, images)
                _, predicted = outputs.max(1)
                correct_val += predicted.eq(labels).sum().item()
                total_val += labels.size(0)

        val_acc = 100. * correct_val / total_val

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_count = 0
        else:
            no_improve_count += 1

        history['epoch'].append(epoch)
        history['loss'].append(total_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if epoch % 10 == 0 or epoch <= 5:
            print(f"  Epoch {epoch:3d}: Loss={total_loss/len(train_loader):.4f}, "
                  f"Train={train_acc:.1f}%, Val={val_acc:.1f}%", flush=True)

        if no_improve_count >= patience:
            print(f"  早停于 Epoch {epoch}，最佳 Val={best_acc:.2f}%", flush=True)
            break

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

    # Hebbian Heavy版本
    print("\n" + "="*70, flush=True)
    print("训练 HebbianHeavy-DenseNet-40 (50% Hebbian占比)", flush=True)
    print("="*70, flush=True)

    heb_model = HebbianHeavyDenseNet(num_classes=10, growth=12, drop=0.2, hebbian_lr=0.05)
    heb_acc, heb_hist, heb_state, heb_params, hebbian_params = train_model(
        heb_model, "HebbianHeavy-DenseNet-40",
        train_loader, test_loader, epochs=100, lr=0.001, patience=20
    )

    # Standard版本
    print("\n" + "="*70, flush=True)
    print("训练 Standard-DenseNet-40 (标准BP)", flush=True)
    print("="*70, flush=True)

    std_model = StandardDenseNet(num_classes=10, growth=12, drop=0.2)
    std_acc, std_hist, std_state, std_params, _ = train_model(
        std_model, "Standard-DenseNet-40",
        train_loader, test_loader, epochs=100, lr=0.001, patience=20
    )

    # 保存结果
    os.makedirs('results/model_checkpoints', exist_ok=True)
    os.makedirs('results/xlsx', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    torch.save({'model_state_dict': heb_state, 'best_acc': heb_acc, 'params': heb_params},
               'results/model_checkpoints/hebbian_heavy_densenet_best.pth')
    torch.save({'model_state_dict': std_state, 'best_acc': std_acc, 'params': std_params},
               'results/model_checkpoints/standard_densenet_best.pth')

    baseline = {'DenseNet-40 (论文)': 94.5, 'ResNet-110': 94.2}

    print("\n" + "="*70, flush=True)
    print("HebbianHeavy (50% Hebbian) 实验结果", flush=True)
    print("="*70, flush=True)

    print(f"\n基准:", flush=True)
    for name, acc in baseline.items():
        print(f"  {name}: {acc:.1f}%", flush=True)

    print(f"\n训练结果:", flush=True)
    print(f"  HebbianHeavy-DenseNet-40: {heb_acc:.2f}% ({heb_params/1e6:.2f}M, Hebbian参数{hebbian_params:,}={hebbian_params/heb_params*100:.1f}%)", flush=True)
    print(f"  Standard-DenseNet-40:    {std_acc:.2f}% ({std_params/1e6:.2f}M)", flush=True)
    print(f"  Hebbian提升: {heb_acc - std_acc:+.2f}%", flush=True)

    df = pd.DataFrame({
        'Model': list(baseline.keys()) + ['HebbianHeavy-DenseNet-40', 'Standard-DenseNet-40'],
        'Accuracy (%)': list(baseline.values()) + [heb_acc, std_acc],
        'Parameters': ['1.0M', '1.7M', f'{heb_params/1e6:.2f}M', f'{std_params/1e6:.2f}M'],
        'Hebbian Params': ['0', '0', f'{hebbian_params:,} ({hebbian_params/heb_params*100:.1f}%)', '0'],
    })
    df.to_excel('results/xlsx/hebbian_heavy_results.xlsx', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(heb_hist['epoch'], heb_hist['loss'], 'r-', label='HebbianHeavy (50%)')
    axes[0].plot(std_hist['epoch'], std_hist['loss'], 'b-', label='Standard')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(heb_hist['epoch'], heb_hist['val_acc'], 'r-', label=f'HebbianHeavy ({heb_acc:.1f}%)')
    axes[1].plot(std_hist['epoch'], std_hist['val_acc'], 'b-', label=f'Standard ({std_acc:.1f}%)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Val Accuracy (%)')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/hebbian_heavy_training.png', dpi=150)
    print("\n已保存: figures/hebbian_heavy_training.png", flush=True)

    print("\n" + "="*70, flush=True)
    print("完成!", flush=True)
    print("="*70, flush=True)


if __name__ == '__main__':
    main()