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

        # Hebbian学习参数
        self.lr = learning_rate
        self.oja_beta = oja_beta  # Oja规则的抑制系数
        self.use_wta = use_wta
        self.wta_ratio = wta_ratio

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
        """前向传播"""
        # 保存前突触活动（用于Hebbian更新）
        self.pre_activity = x

        # 标准卷积
        out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding)

        # 保存后突触活动
        self.post_activity = out

        return out

    def hebbian_update(self):
        """
        Hebbian权重更新

        公式: Δw = lr * pre * post - beta * post^2 * w (Oja规则)

        核心思想：
        - 前突触和后突触同时活跃时，权重增强
        - Oja项防止权重无限增长
        """
        if self.pre_activity is None or self.post_activity is None:
            return

        # 计算Hebbian更新量
        # pre: [B, C_in, H, W]
        # post: [B, C_out, H', W']

        # 为了计算权重更新，需要对空间维度求平均
        pre_mean = self.pre_activity.mean(dim=(2, 3))  # [B, C_in]
        post_mean = self.post_activity.mean(dim=(2, 3))  # [B, C_out]

        # 计算外积形式的更新
        # weight: [C_out, C_in, K, K]

        delta_w = torch.zeros_like(self.weight)

        for b in range(pre_mean.shape[0]):
            # 外积: [C_in, 1] @ [1, C_out] -> [C_in, C_out]
            hebbian_term = torch.outer(post_mean[b], pre_mean[b])  # [C_out, C_in]

            # Oja抑制项
            oja_term = self.oja_beta * (post_mean[b]**2).mean() * self.weight

            delta_w += hebbian_term.unsqueeze(-1).unsqueeze(-1)

        delta_w = delta_w / pre_mean.shape[0] - oja_term

        # 更新权重
        self.weight.data += self.lr * delta_w

        # 可选：竞争机制（WTA）
        if self.use_wta:
            self._wta竞争()

    def _wta竞争(self):
        """
        Winner-Takes-All 竞争机制

        只保留最强的连接，增强特征稀疏性
        模拟神经科学中的侧抑制机制
        """
        with torch.no_grad():
            # 计算每个输出通道的权重范数
            weight_norm = self.weight.data.pow(2).sum(dim=(1, 2, 3))  # [C_out]

            # 找出要保留的通道数
            n_keep = max(1, int(self.out_channels * self.wta_ratio))

            # 确定 winners
            _, winners = torch.topk(weight_norm, n_keep)

            # 抑制非 winners（乘以小于1的因子）
            mask = torch.ones_like(self.weight.data)
            for w in winners:
                mask[w] = 0.0

            # 弱化非 winners
            self.weight.data *= (1 - 0.1 * mask)

    def get_weight_norm(self) -> torch.Tensor:
        """获取权重范数（用于监控）"""
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