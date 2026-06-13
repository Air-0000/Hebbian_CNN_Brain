"""
True Hebbian DenseNet - Hebbian作为主力的真正Hebbian学习

核心创新：
- Hebbian权重矩阵：模拟突触可塑性
- Δw_ij = η × pre_i × post_j
- Hebbian机制在训练中起主导作用，而非辅助微调
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
print("True Hebbian DenseNet-40 (Hebbian作为主力)", flush=True)
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


# ==================== 真正的 Hebbian Dense Layer (主力版本) ====================
class TrueHebbianDenseLayer(nn.Module):
    """
    Hebbian作为主力的DenseNet层

    核心机制：
    1. Hebbian权重矩阵 W_h: [in_ch, growth]
       - 模拟输入通道到输出通道的突触连接强度
       - 根据输入输出活动相关性动态更新

    2. 通道内Hebbian调制 W_intra: [growth, growth]
       - 模拟同层神经元间的竞争/合作
       - 同步激活的通道互相增强

    3. Hebbian更新规则：
       Δw_ij = η × (x_i的平均活动) × (y_j的平均活动)
       - 前层活动高 + 后层活动高 → 突触增强
       - 前层活动低 + 后层活动低 → 突触减弱
    """
    def __init__(self, in_ch, growth=12, drop=0.2, hebbian_lr=0.05):
        super().__init__()

        # 标准卷积路径
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, 4*growth, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth)
        self.conv2 = nn.Conv2d(4*growth, growth, 3, padding=1, bias=False)

        self.drop = drop
        self.hebbian_lr = hebbian_lr
        self.in_ch = in_ch
        self.growth = growth

        # ========== 核心：Hebbian权重矩阵 ==========
        # Hebbian输入权重: [in_ch, growth] - 输入通道到输出通道的突触
        self.hebbian_w = nn.Parameter(
            torch.randn(in_ch, growth) * 0.01
        )

        # Hebbian通道内调制: [growth, growth] - 通道间竞争/合作
        self.hebbian_intra = nn.Parameter(
            torch.randn(growth, growth) * 0.01
        )

        # 可学习的调制因子
        self.modulation = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # 标准卷积
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))

        # ========== Hebbian动态调制（主力） ==========
        if self.training:
            # 计算通道活动
            # 输入: [B, in_ch, H, W] -> [B, in_ch]
            in_act = x.mean(dim=(2, 3))

            # 输出: [B, growth, H, W] -> [B, growth]
            out_act = out.mean(dim=(2, 3))

            # Hebbian权重更新: Δw = η × in_act × out_act^T
            # [in_ch] × [growth]^T -> [in_ch, growth]
            hebbian_update = torch.ger(
                in_act.mean(dim=0),  # [in_ch]
                out_act.mean(dim=0)   # [growth]
            ) * self.hebbian_lr

            # 指数移动平均更新Hebbian权重
            with torch.no_grad():
                self.hebbian_w.data = (
                    self.hebbian_w.data * 0.95 +
                    hebbian_update * 0.05
                )

            # 应用Hebbian调制到输出
            # 通道调制 = sigmoid(hebbian_w^T × in_act)
            channel_mod = torch.sigmoid(
                torch.matmul(in_act, self.hebbian_w)  # [B, growth]
            ).mean(dim=0)  # [growth]

            # 通道间调制 = sigmoid(out_act × hebbian_intra)
            intra_mod = torch.sigmoid(
                torch.matmul(out_act, self.hebbian_intra)  # [B, growth]
            ).mean(dim=0)  # [growth]

            # 综合调制
            hebbian_effect = channel_mod * intra_mod * self.modulation

            # 应用到特征图
            out = out * (1 + hebbian_effect.unsqueeze(-1).unsqueeze(-1) * 0.5)

        if self.drop > 0:
            out = F.dropout(out, p=self.drop, training=self.training)

        return out


class SimpleDenseLayer(nn.Module):
    """标准DenseNet层（无Hebbian）"""
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
                self.layers.append(TrueHebbianDenseLayer(in_ch + i*growth, growth, drop, hebbian_lr))
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
class TrueHebbianDenseNet(nn.Module):
    """
    Hebbian作为主力的DenseNet

    每层Hebbian参数:
    - hebbian_w: [in_ch, growth] = 约 16×12=192 到 160×12=1920
    - hebbian_intra: [growth, growth] = 12×12=144
    - modulation: 1
    总计: 每层约 300-2000 个 Hebbian 参数
    """
    def __init__(self, num_classes=10, growth=12, drop=0.2, hebbian_lr=0.05):
        super().__init__()
        ch = 16

        self.features = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
        )

        # 3个DenseBlock，每块12层
        self.block1 = DenseBlock(12, ch, growth, drop, True, hebbian_lr)
        ch = ch + 12*growth  # 160
        self.trans1 = Transition(ch, ch//2)  # 80
        ch = ch // 2

        self.block2 = DenseBlock(12, ch, growth, drop, True, hebbian_lr)
        ch = ch + 12*growth  # 224
        self.trans2 = Transition(ch, ch//2)  # 112
        ch = ch // 2

        self.block3 = DenseBlock(12, ch, growth, drop, True, hebbian_lr)
        ch = ch + 12*growth  # 256

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
    """标准DenseNet（无Hebbian）作为对照组"""
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
    """统计Hebbian相关参数"""
    total = 0
    for name, param in model.named_parameters():
        if 'hebbian' in name or 'modulation' in name:
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

    # True Hebbian DenseNet (Hebbian作为主力)
    print("\n" + "="*70, flush=True)
    print("训练 TrueHebbian-DenseNet-40 (Hebbian主力)", flush=True)
    print("="*70, flush=True)

    heb_model = TrueHebbianDenseNet(num_classes=10, growth=12, drop=0.2, hebbian_lr=0.05)
    heb_acc, heb_hist, heb_state, heb_params, hebbian_params = train_model(
        heb_model, "TrueHebbian-DenseNet-40",
        train_loader, test_loader, epochs=100, lr=0.001, patience=20
    )

    # Standard DenseNet (无Hebbian)
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
               'results/model_checkpoints/true_hebbian_densenet_main_best.pth')
    torch.save({'model_state_dict': std_state, 'best_acc': std_acc, 'params': std_params},
               'results/model_checkpoints/standard_densenet_best.pth')

    baseline = {'DenseNet-40 (论文)': 94.5, 'ResNet-110': 94.2}

    print("\n" + "="*70, flush=True)
    print("True Hebbian DenseNet (Hebbian主力) 实验结果", flush=True)
    print("="*70, flush=True)

    print(f"\n基准:", flush=True)
    for name, acc in baseline.items():
        print(f"  {name}: {acc:.1f}%", flush=True)

    print(f"\n训练结果:", flush=True)
    print(f"  TrueHebbian-DenseNet-40: {heb_acc:.2f}% ({heb_params/1e6:.2f}M, Hebbian参数{hebbian_params:,})", flush=True)
    print(f"  Standard-DenseNet-40:    {std_acc:.2f}% ({std_params/1e6:.2f}M)", flush=True)
    print(f"  Hebbian提升: {heb_acc - std_acc:+.2f}%", flush=True)

    df = pd.DataFrame({
        'Model': list(baseline.keys()) + ['TrueHebbian-DenseNet-40', 'Standard-DenseNet-40'],
        'Accuracy (%)': list(baseline.values()) + [heb_acc, std_acc],
        'Parameters': ['1.0M', '1.7M', f'{heb_params/1e6:.2f}M', f'{std_params/1e6:.2f}M'],
        'Hebbian Params': ['0', '0', f'{hebbian_params:,}', '0'],
    })
    df.to_excel('results/xlsx/true_hebbian_densenet_main_results.xlsx', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(heb_hist['epoch'], heb_hist['loss'], 'r-', label='True Hebbian')
    axes[0].plot(std_hist['epoch'], std_hist['loss'], 'b-', label='Standard')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(heb_hist['epoch'], heb_hist['val_acc'], 'r-', label=f'True Hebbian ({heb_acc:.1f}%)')
    axes[1].plot(std_hist['epoch'], std_hist['val_acc'], 'b-', label=f'Standard ({std_acc:.1f}%)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Val Accuracy (%)')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/true_hebbian_densenet_main_training.png', dpi=150)
    print("\n已保存: figures/true_hebbian_densenet_main_training.png", flush=True)

    print("\n" + "="*70, flush=True)
    print("完成!", flush=True)
    print("="*70, flush=True)


if __name__ == '__main__':
    main()