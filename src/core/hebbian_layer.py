"""
Hebbian卷积层 - 结合神经科学原理的CNN层实现

核心原理：
1. Hebbian规则: "一起放电的神经元会连在一起"
2. Oja规则: 防止权重无限增长
3. 竞争机制: Winner-Takes-All 产生稀疏特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class HebbianConv2d(nn.Module):
    """
    Hebbian卷积层

    实现思路：
    - 前向传播：标准卷积
    - 更新规则：用Hebbian规则而非BP更新权重
    - 竞争机制：保留最强连接
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        learning_rate: float = 0.01,
        oja_beta: float = 0.001,
        use_wta: bool = True,
        wta_ratio: float = 0.3,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Hebbian学习与可塑性参数
        self.lr = learning_rate
        self.oja_beta = oja_beta  # Oja规则抑制系数
        self.use_wta = use_wta
        self.wta_ratio = wta_ratio

        # BCM (Bienenstock-Cooper-Munro) 可塑性阈值
        self.register_buffer("bcm_threshold", torch.ones(out_channels) * 0.1)
        self.bcm_tau = 0.95  # 滑动平均衰减率

        # 皮层侧向抑制矩阵 (Cortical Lateral Inhibition)
        self.register_buffer("lateral_inhibition_weights", 
                             torch.eye(out_channels) * 0.0 + 
                             (1.0 - torch.eye(out_channels)) * (0.15 / max(1, out_channels - 1)))

        # 初始化权重（使用Kaiming初始化）
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # 用于Hebbian学习的中间变量
        self.pre_activity = None
        self.post_activity = None

        # 可学习参数（用于归一化）
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播（包含皮层侧向抑制）"""
        self.pre_activity = x

        # 标准卷积
        out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding)

        # 皮层侧向抑制 (Lateral Inhibition across channels)
        if self.use_wta and self.training:
            # out: [B, C_out, H, W]
            B, C, H, W = out.shape
            out_flat = out.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
            # y_inhibited = ReLU(y - alpha * W_lat @ y)
            inhibition = out_flat @ self.lateral_inhibition_weights.T
            out_inhibited = F.relu(out_flat - 0.2 * inhibition)
            out = out_inhibited.reshape(B, H, W, C).permute(0, 3, 1, 2)

        self.post_activity = out
        return out

    def hebbian_update(self, mode: str = "spatial_bcm"):
        """
        生物可塑性权重更新

        支持两种模式：
        1. 'spatial_bcm': 空间感受野局部滑动阈值 BCM 规则（真正的二维空间突触可塑性）
           - 利用 F.unfold 提取局部感受野 patch
           - 计算滑动阈值 theta_M = E[post^2]
           - delta_w = lr * post * (post - theta_M) * pre_unfold
        2. 'oja': Sanger/Oja 主成分分析规则
           - delta_w = lr * (post * pre - beta * post^2 * w)
        """
        if self.pre_activity is None or self.post_activity is None:
            return

        B, C_in, H_in, W_in = self.pre_activity.shape
        _, C_out, H_out, W_out = self.post_activity.shape

        with torch.no_grad():
            # 1. 展开前突触局部感受野 patch: [B, C_in * K * K, L]
            pre_unfold = F.unfold(
                self.pre_activity,
                kernel_size=self.kernel_size,
                padding=self.padding,
                stride=self.stride
            )
            L = pre_unfold.shape[-1]

            # 2. 展平后突触活动: [B, C_out, L]
            post_flat = self.post_activity.reshape(B, C_out, -1)
            if post_flat.shape[-1] != L:
                post_flat = F.interpolate(post_flat, size=L, mode="nearest")

            if mode == "spatial_bcm":
                # --- BCM 理论滑动阈值更新 ---
                # 更新自适应阈值 theta_M: [C_out]
                current_activity_sq = (post_flat ** 2).mean(dim=(0, 2))
                self.bcm_threshold.copy_(
                    self.bcm_tau * self.bcm_threshold + (1.0 - self.bcm_tau) * current_activity_sq
                )

                # BCM 调节函数 phi(y, theta) = y * (y - theta_M)
                phi = post_flat * (post_flat - self.bcm_threshold.view(1, C_out, 1))  # [B, C_out, L]

                # 外积关联: [C_out, C_in * K * K]
                # delta_w_flat = (phi @ pre_unfold.transpose(1, 2)).mean(dim=0) / L
                delta_w_flat = torch.bmm(phi, pre_unfold.transpose(1, 2)).mean(dim=0) / float(L)
                delta_w = delta_w_flat.view(C_out, C_in, self.kernel_size, self.kernel_size)

                # 衰减项稳定权重
                decay = 0.001 * self.weight.data
                self.weight.data += self.lr * (delta_w - decay)

            else:
                # --- Oja 规则更新 ---
                # post_flat: [B, C_out, L], pre_unfold: [B, C_in*K*K, L]
                delta_w_flat = torch.bmm(post_flat, pre_unfold.transpose(1, 2)).mean(dim=0) / float(L)
                delta_w = delta_w_flat.view(C_out, C_in, self.kernel_size, self.kernel_size)

                # Oja 归一化二次项
                oja_term = self.oja_beta * (post_flat ** 2).mean() * self.weight.data
                self.weight.data += self.lr * (delta_w - oja_term)

            # 3. 突触权重范数边界约束 (防止数值溢出)
            weight_norms = torch.norm(self.weight.data.view(C_out, -1), dim=1, keepdim=True)
            scale = torch.clamp(weight_norms, min=1.0)
            self.weight.data /= scale.view(C_out, 1, 1, 1)

    def _wta竞争(self):
        """Winner-Takes-All 稀疏化"""
        with torch.no_grad():
            weight_norm = self.weight.data.pow(2).sum(dim=(1, 2, 3))
            n_keep = max(1, int(self.out_channels * self.wta_ratio))
            _, winners = torch.topk(weight_norm, n_keep)
            mask = torch.ones_like(self.weight.data)
            for w in winners:
                mask[w] = 0.0
            self.weight.data *= (1 - 0.1 * mask)

    def get_weight_norm(self) -> torch.Tensor:
        """获取各通道权重范数"""
        return self.weight.data.pow(2).sum(dim=(1, 2, 3))


class HebbianConvBlock(nn.Module):
    """
    Hebbian卷积块 - 包含卷积、BN、激活

    设计参考VGG结构，但使用Hebbian学习
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_pool: bool = True,
        hebbian_lr: float = 0.01,
    ):
        super().__init__()

        self.conv = HebbianConv2d(
            in_channels, out_channels, kernel_size,
            stride, padding, learning_rate=hebbian_lr
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2, 2) if use_pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.relu(self.bn(x))
        x = self.pool(x)
        return x


class HebbianNetwork(nn.Module):
    """
    完整Hebbian卷积网络

    结构: Conv -> Pool -> Conv -> Pool -> FC
    训练方式: Hebbian更新 + 最终分类层BP微调
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        hebbian_channels: list = [16, 32],
        fc_hidden: int = 128,
    ):
        super().__init__()

        # Hebbian卷积层
        self.features = nn.Sequential(
            HebbianConvBlock(1, 16, use_pool=True, hebbian_lr=0.01),
            HebbianConvBlock(16, 32, use_pool=True, hebbian_lr=0.01),
        )

        # 特征图大小 (假设输入28x28)
        self.feature_size = 32 * 7 * 7

        # 分类器（用BP微调）
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, fc_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden, num_classes),
        )

        # Hebbian层列表（用于更新）
        self.hebbian_layers = []
        for m in self.modules():
            if isinstance(m, HebbianConv2d):
                self.hebbian_layers.append(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def hebbian_update_all(self):
        """更新所有Hebbian层"""
        for layer in self.hebbian_layers:
            layer.hebbian_update()

    def get_feature_similarity(self) -> torch.Tensor:
        """
        计算特征相似度（监控学习效果）
        高相似度 = 特征同质化（不好）
        低相似度 = 特征多样化（好）
        """
        weight_stack = torch.stack([
            layer.weight.data.flatten(start_dim=1)
            for layer in self.hebbian_layers
        ])  # [n_layers, n_params]

        # 计算余弦相似度矩阵
        norm = weight_stack / weight_stack.norm(dim=1, keepdim=True)
        similarity = norm @ norm.T
        return similarity


def create_hebbian_network(config: dict) -> HebbianNetwork:
    """工厂函数：根据配置创建网络"""
    return HebbianNetwork(
        input_channels=config.get('input_channels', 1),
        num_classes=config.get('num_classes', 10),
        hebbian_channels=config.get('hebbian_channels', [16, 32]),
        fc_hidden=config.get('fc_hidden', 128),
    )