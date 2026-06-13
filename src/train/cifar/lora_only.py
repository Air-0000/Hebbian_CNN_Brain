"""
LoRA-BioHebbianNet v3 - 直接LoRA微调
跳过全量微调，只做LoRA微调
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
print("LoRA-BioHebbianNet v3 - 直接LoRA微调", flush=True)
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


class NeuroBlock(nn.Module):
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
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            NeuroBlock(3, 64, use_neuro=True),
            nn.MaxPool2d(2),
            NeuroBlock(64, 128, use_neuro=True),
            nn.MaxPool2d(2),
            NeuroBlock(128, 256, use_neuro=True),
            nn.MaxPool2d(2),
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


class LoRAConv(nn.Module):
    """LoRA卷积: 原始权重冻结，只训练LoRA"""
    def __init__(self, orig_conv, rank=4, alpha=16):
        super().__init__()
        self.orig_conv = orig_conv
        self.scaling = alpha / rank

        # 冻结原始权重
        for p in self.orig_conv.parameters():
            p.requires_grad = False

        # LoRA: 1x1卷积
        self.lora = nn.Conv2d(
            orig_conv.in_channels, orig_conv.out_channels, 1,
            stride=orig_conv.stride, padding=0
        )

        # Hebbian
        self.has_hebbian = hasattr(orig_conv, 'scale') and orig_conv.scale is not None
        if self.has_hebbian:
            self.scale = nn.Parameter(orig_conv.scale.data.clone())
            self.pv = nn.Parameter(orig_conv.pv.data.clone())

    def forward(self, x):
        out = self.orig_conv(x) + self.lora(x) * self.scaling
        if self.has_hebbian:
            out = out * (1 + self.scale)
            pv_factor = torch.sigmoid(self.pv * (out.mean() - 0.3))
            out = out * (1 - pv_factor * 0.3)
        return out


class LoRALinear(nn.Module):
    """LoRA全连接"""
    def __init__(self, orig_linear, rank=4, alpha=16):
        super().__init__()
        self.orig_linear = orig_linear
        self.scaling = alpha / rank

        for p in self.orig_linear.parameters():
            p.requires_grad = False

        self.lora_A = nn.Parameter(torch.randn(orig_linear.in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(rank, orig_linear.out_features) * 0.01)

    def forward(self, x):
        return self.orig_linear(x) + (x @ self.lora_A @ self.lora_B) * self.scaling


def create_lora_model(original_model, rank=4, alpha=16):
    """创建LoRA版本"""
    model = BioHebbianNetV3()
    model.load_state_dict(original_model.state_dict())

    for i, module in enumerate(model.features):
        if isinstance(module, NeuroBlock):
            model.features[i].conv1 = LoRAConv(module.conv1, rank, alpha)
            model.features[i].conv2 = LoRAConv(module.conv2, rank, alpha)

    model.classifier[2] = LoRALinear(model.classifier[2], rank, alpha)
    model.classifier[5] = LoRALinear(model.classifier[5], rank, alpha)

    return model


def predict_with_tta(model, images):
    model.eval()
    with torch.no_grad():
        out1 = torch.softmax(model(images), dim=1)
        out2 = torch.softmax(model(torch.flip(images, [3])), dim=1)
        return (out1 + out2) / 2


def train_model(model, name, train_loader, test_loader, epochs=30, lr=0.01):
    print(f"\n训练 {name}...", flush=True)
    model = model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print(f"  可训练: {sum(p.numel() for p in trainable):,}", flush=True)
    print(f"  冻结: {frozen:,}", flush=True)
    print(f"  比例: {sum(p.numel() for p in trainable)/total*100:.1f}%", flush=True)

    optimizer = optim.AdamW(trainable, lr=lr, weight_decay=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = LabelSmoothingCE(smoothing=0.1)

    best_acc = 0
    best_state = None
    history = {'epoch': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, epochs + 1):
        model.train()
        correct, total = 0, 0

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
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

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
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"  Epoch {epoch:2d}: Train={train_acc:.1f}%, Val={val_acc:.1f}%", flush=True)

    print(f"  最佳: {best_acc:.2f}%", flush=True)
    return best_acc, history, best_state


def main():
    pretrained_path = 'results/model_checkpoints/bio_hebbian_v3_best.pth'

    print(f"\n加载预训练: {pretrained_path}", flush=True)
    checkpoint = torch.load(pretrained_path, weights_only=False)
    pretrained_state = checkpoint['model_state_dict']
    print(f"✓ 预训练准确率: {checkpoint.get('best_acc', 0):.2f}%", flush=True)

    original_model = BioHebbianNetV3()
    original_model.load_state_dict(pretrained_state)

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

    # 直接LoRA微调
    print("\n" + "="*70, flush=True)
    print("LoRA微调 (冻结原始权重，只训练LoRA)", flush=True)
    print("="*70, flush=True)

    lora_model = create_lora_model(original_model, rank=4, alpha=16)

    lora_acc, history, lora_state = train_model(
        lora_model, "LoRA-Finetune",
        train_loader, test_loader, epochs=30, lr=0.01
    )

    os.makedirs('results/model_checkpoints', exist_ok=True)
    torch.save({'model_state_dict': lora_state, 'best_acc': lora_acc},
               'results/model_checkpoints/lora_only_finetune.pth')
    print(f"\n已保存: results/model_checkpoints/lora_only_finetune.pth", flush=True)

    # 结果
    print("\n" + "="*70, flush=True)
    print("结果", flush=True)
    print("="*70, flush=True)
    print(f"  预训练起点: {checkpoint.get('best_acc', 0):.2f}%", flush=True)
    print(f"  LoRA微调后: {lora_acc:.2f}%", flush=True)
    print(f"  提升: {lora_acc - checkpoint.get('best_acc', 0):+.2f}%", flush=True)

    # 保存
    os.makedirs('results/xlsx', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    pd.DataFrame({
        'Model': ['预训练起点', 'LoRA微调'],
        'Accuracy (%)': [checkpoint.get('best_acc', 0), lora_acc],
    }).to_excel('results/xlsx/lora_results.xlsx', index=False)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['epoch'], history['train_acc'], 'r-', label='Train')
    plt.plot(history['epoch'], history['val_acc'], 'b-', label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('LoRA Finetune')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.bar(['预训练起点', 'LoRA微调'], [checkpoint.get('best_acc', 0), lora_acc], color=['steelblue', 'coral'])
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('figures/lora_results.png', dpi=150)
    print("\n完成!", flush=True)


if __name__ == '__main__':
    main()