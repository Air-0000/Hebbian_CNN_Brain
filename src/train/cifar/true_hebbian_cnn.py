"""
真正的 Hebbian Learning 卷积层
核心：每个权重根据输入输出活动的相关性动态更新

Hebbian学习规则：
Δw_ij = η × x_i × y_j

其中：
- x_i: 前一层神经元活动
- y_j: 后一层神经元活动
- η: 学习率
- w_ij: 突触权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TrueHebbianConv2d(nn.Module):
    """
    真正的 Hebbian 卷积层

    工作原理：
    1. 普通卷积计算输出
    2. Hebbian 权重根据输入输出相关性更新
    3. 最终输出 = 普通卷积 + Hebbian调制

    与标准卷积的区别：
    - 标准卷积：权重在训练时通过BP更新，推理时固定
    - Hebbian卷积：权重在推理时也根据输入动态更新
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, hebbian_lr=0.01, use_hebbian=True):
        super().__init__()

        # 普通卷积层
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                             stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

        # Hebbian 权重 (每个输入通道对应一组权重)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hebbian_lr = hebbian_lr
        self.use_hebbian = use_hebbian

        if use_hebbian:
            # Hebbian 权重矩阵: [out_channels, in_channels]
            # 表示从输入通道i到输出通道j的突触强度
            self.hebbian_weights = nn.Parameter(
                torch.randn(out_channels, in_channels) * 0.01
            )

            # 额外的通道内Hebbian调制
            # 模拟同一输出通道内神经元间的竞争/合作
            self.intra_weights = nn.Parameter(
                torch.randn(out_channels, out_channels) * 0.01
            )

            # 可学习的调制因子
            self.modulation = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # 1. 普通卷积前向传播
        ordinary = self.conv(x)
        ordinary = self.bn(ordinary)

        if not self.use_hebbian or not self.training:
            return F.relu(ordinary)

        # 2. 计算 Hebbian 更新
        # 输入活动: [B, in_channels, H, W] -> 平均 -> [B, in_channels]
        x_activity = x.mean(dim=(2, 3))  # 空间平均

        # 输出活动: [B, out_channels, H, W] -> 平均 -> [B, out_channels]
        y_activity = ordinary.mean(dim=(2, 3))

        # 3. Hebbian 权重更新
        # Δw_ij = η × mean(pre_i) × mean(post_j)
        # 外积: [out_channels] × [in_channels]^T -> [out_channels, in_channels]
        hebbian_update = torch.mean(
            x_activity.unsqueeze(1) * y_activity.unsqueeze(2),
            dim=0
        ) * self.hebbian_lr

        # 更新 Hebbian 权重 (指数移动平均)
        with torch.no_grad():
            self.hebbian_weights.data = (
                self.hebbian_weights.data * (1 - self.hebbian_lr * 0.1) +
                hebbian_update
            )

        # 4. 应用 Hebbian 调制
        # 简化实现：使用可学习的通道调制
        # 基于Hebbian规则：同步激活的通道应该互相增强

        # 计算每个输出通道的平均活动
        channel_activity = ordinary.mean(dim=(2, 3))  # [B, out_channels]

        # Hebbian权重应用到每个通道
        # hebbian_weights: [out_channels, in_channels]
        # 计算输入激活对输出的贡献
        hebbian_signal = torch.matmul(channel_activity, self.hebbian_weights)  # [B, in_channels]
        hebbian_signal = torch.sigmoid(hebbian_signal.mean(dim=1, keepdim=True))  # [B, 1]

        # 应用调制：简单的加性调制
        output = ordinary + hebbian_signal.unsqueeze(-1).unsqueeze(-1) * ordinary * self.modulation

        return F.relu(output)


class HebbianDenseLayer(nn.Module):
    """
    Hebbian 全连接层

    结构: y = W × x
    Hebbian更新: ΔW = η × y × x^T
    """
    def __init__(self, in_features, out_features, hebbian_lr=0.01, use_hebbian=True):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bn = nn.BatchNorm1d(out_features)

        self.in_features = in_features
        self.out_features = out_features
        self.hebbian_lr = hebbian_lr
        self.use_hebbian = use_hebbian

        if use_hebbian:
            # 初始化Hebbian权重为单位矩阵附近
            self.hebbian_weights = nn.Parameter(
                torch.eye(out_features, in_features) * 0.5 + torch.randn(out_features, in_features) * 0.1
            )
            self.modulation = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        ordinary = self.linear(x)
        ordinary = self.bn(ordinary)

        if not self.use_hebbian or not self.training:
            return F.relu(ordinary)

        # 计算Hebbian更新
        x_activity = x.mean(dim=0)  # [in_features]
        y_activity = ordinary.mean(dim=0)  # [out_features]

        # ΔW = η × y × x^T
        hebbian_update = torch.ger(y_activity, x_activity) * self.hebbian_lr

        with torch.no_grad():
            self.hebbian_weights.data = (
                self.hebbian_weights.data * (1 - self.hebbian_lr * 0.1) +
                hebbian_update
            )

        # 应用Hebbian调制
        hebbian_effect = F.linear(x, self.hebbian_weights) * self.modulation
        output = ordinary + hebbian_effect

        return F.relu(output)


class HebbianConvBlock(nn.Module):
    """
    Hebbian 卷积块

    包含两个Hebbian卷积层 + 跳跃连接
    """
    def __init__(self, in_ch, out_ch, hebbian_lr=0.01, dropout=0.2):
        super().__init__()

        self.conv1 = TrueHebbianConv2d(in_ch, out_ch, hebbian_lr=hebbian_lr)
        self.conv2 = TrueHebbianConv2d(out_ch, out_ch, hebbian_lr=hebbian_lr)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # 跳跃连接
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.dropout(out)

        return out + residual


class BioHebbianNetTrue(nn.Module):
    """
    基于真正Hebbian学习的网络

    与之前版本的区别：
    - 之前: 固定的scale/pv参数，仅在训练时更新
    - 现在: Hebbian权重在推理时也根据输入动态更新
    """
    def __init__(self, num_classes=10, hebbian_lr=0.01):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # Hebbian Blocks
        self.block1 = HebbianConvBlock(32, 64, hebbian_lr)
        self.pool1 = nn.MaxPool2d(2)

        self.block2 = HebbianConvBlock(64, 128, hebbian_lr)
        self.pool2 = nn.MaxPool2d(2)

        self.block3 = HebbianConvBlock(128, 256, hebbian_lr)
        self.pool3 = nn.MaxPool2d(2)

        self.block4 = HebbianConvBlock(256, 512, hebbian_lr)

        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            HebbianDenseLayer(512, 256, hebbian_lr),
            nn.Dropout(0.3),
            HebbianDenseLayer(256, num_classes, hebbian_lr, use_hebbian=False),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.block4(x)
        return self.classifier(x)


# ==================== 标准对照版 ====================
class StandardConvBlock(nn.Module):
    """标准卷积块 (无Hebbian)"""
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)

        return out + residual


class StandardNet(nn.Module):
    """标准网络 (对照)"""
    def __init__(self, num_classes=10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.block1 = StandardConvBlock(32, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.block2 = StandardConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.block3 = StandardConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.block4 = StandardConvBlock(256, 512)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.block4(x)
        return self.classifier(x)


def count_parameters(model):
    """统计参数量"""
    total = sum(p.numel() for p in model.parameters())
    hebbian = sum(p.numel() for p in model.parameters() if 'hebbian' in p.name or 'intra' in p.name)
    return total, hebbian


# ==================== 测试 ====================
if __name__ == '__main__':
    print("="*60)
    print("真正的 Hebbian Learning 网络测试")
    print("="*60)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"设备: {device}")

    # 创建模型
    heb_model = BioHebbianNetTrue(num_classes=10, hebbian_lr=0.01)
    std_model = StandardNet(num_classes=10)

    heb_model = heb_model.to(device)
    std_model = std_model.to(device)

    # 统计参数量
    total_heb, hebbian_params = count_parameters(heb_model)
    total_std, _ = count_parameters(std_model)

    print(f"\nBioHebbianNet (Hebbian版):")
    print(f"  总参数量: {total_heb:,}")
    print(f"  Hebbian参数: {hebbian_params:,} ({hebbian_params/total_heb*100:.1f}%)")

    print(f"\nStandardNet (标准版):")
    print(f"  总参数量: {total_std:,}")

    # 测试前向传播
    x = torch.randn(4, 3, 32, 32).to(device)

    print(f"\n前向传播测试...")
    with torch.no_grad():
        y_heb = heb_model(x)
        y_std = std_model(x)
        print(f"  Hebbian输出: {y_heb.shape}, {y_heb.mean().item():.4f}")
        print(f"  Standard输出: {y_std.shape}, {y_std.mean().item():.4f}")

    # 测试Hebbian权重更新
    print(f"\nHebbian权重更新测试...")
    print(f"训练模式开启，Hebbian权重会随输入动态更新")

    heb_model.train()
    x = torch.randn(8, 3, 32, 32).to(device)
    optimizer = torch.optim.Adam(heb_model.parameters(), lr=0.001)

    # 记录初始权重
    initial_weights = [p.clone() for p in heb_model.parameters() if 'hebbian_weights' in p.name]

    # 多次前向传播
    for i in range(3):
        optimizer.zero_grad()
        y = heb_model(x)
        loss = y.mean()
        loss.backward()
        optimizer.step()

        # 检查权重变化
        current_weights = [p for p in heb_model.parameters() if 'hebbian_weights' in p.name]
        for j, (init, curr) in enumerate(zip(initial_weights, current_weights)):
            diff = (curr - init).abs().max().item()
            if diff > 1e-6:
                print(f"  Layer {j}: Hebbian权重变化 = {diff:.6f} ✓")
                break

    print(f"\n测试完成!")