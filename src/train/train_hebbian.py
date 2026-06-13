"""
Hebbian学习训练脚本

与反向传播(BP)训练对比：
- Hebbian: 无监督、局部更新、生物学plausible
- BP: 有监督、全局更新、效率高但生物学不plausible

训练流程：
1. 分子层初始化
2. 突触层Hebbian预训练
3. 网络层Lindsay模型训练
4. 系统层CNN特征提取
5. 分类器微调
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from layer_implementation import (
    MolecularLayer, SynapticLayer, LindsayNetwork,
    CNNSystemLayer, HebbianCNN, LayerVisualizer
)


# ============================================================
# 数据加载
# ============================================================

def load_mnist(batch_size: int = 64):
    """加载MNIST数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    train_dataset = datasets.MNIST(
        root='./data', train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, transform=transform, download=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# ============================================================
# Hebbian训练
# ============================================================

class HebbianTrainer:
    """
    Hebbian学习训练器

    训练策略：
    1. Hebbian预训练（无监督）- 学习特征表示
    2. BP微调（有监督）- 优化分类性能
    """

    def __init__(
        self,
        n_classes: int = 10,
        hebbian_epochs: int = 5,
        finetune_epochs: int = 10,
        learning_rate: float = 0.001,
    ):
        self.n_classes = n_classes
        self.hebbian_epochs = hebbian_epochs
        self.finetune_epochs = finetune_epochs
        self.lr = learning_rate

        # 创建模型
        self.model = HebbianCNN(n_classes=n_classes)

        # 分子层
        self.molecular = MolecularLayer(n_synapses=1000)

        # 优化器（仅用于微调阶段）
        self.optimizer = optim.Adam(self.model.system.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

        # 记录
        self.history = {
            'hebbian': {'loss': [], 'acc': []},
            'finetune': {'loss': [], 'acc': []},
        }

    def extract_features(self, images: torch.Tensor) -> np.ndarray:
        """使用Hebbian层提取特征"""
        with torch.no_grad():
            features = self.model.system.conv1(images)
            features = torch.nn.functional.relu(features)
            features = self.model.system.pool(features)
            features = self.model.system.conv2(features)
            features = torch.nn.functional.relu(features)
            features = self.model.system.pool(features)
            return features.numpy()

    def train_hebbian(
        self,
        train_loader: DataLoader,
        visualize_callback=None,
    ):
        """
        Hebbian无监督预训练阶段

        核心：用Hebbian规则更新卷积核权重
        """
        print("\n" + "=" * 50)
        print("阶段1: Hebbian无监督预训练")
        print("=" * 50)

        # 初始化Hebbian层
        conv1_weight_orig = self.model.system.conv1.weight.data.clone()
        conv2_weight_orig = self.model.system.conv2.weight.data.clone()

        for epoch in range(self.hebbian_epochs):
            total_loss = 0
            batch_count = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                # Hebbian更新卷积层
                self._hebbian_update_conv(
                    images,
                    self.model.system.conv1,
                    conv1_weight_orig
                )
                self._hebbian_update_conv(
                    images,
                    self.model.system.conv2,
                    conv2_weight_orig
                )

                # 计算重建误差（作为损失）
                with torch.no_grad():
                    features = self.extract_features(images)
                    recon = self._simple_reconstruct(features)
                    loss = np.mean((features - recon) ** 2)
                    total_loss += loss
                    batch_count += 1

                if batch_idx % 100 == 0:
                    print(f"  Epoch {epoch+1}/{self.hebbian_epochs} "
                          f"[Batch {batch_idx}/{len(train_loader)}] "
                          f"Loss: {loss:.4f}")

                    if visualize_callback:
                        visualize_callback(self.model, epoch)

            avg_loss = total_loss / batch_count
            self.history['hebbian']['loss'].append(avg_loss)
            print(f"  Epoch {epoch+1} 完成, 平均损失: {avg_loss:.4f}")

        print("\n  Hebbian预训练完成!")
        print(f"  Conv1权重变化: {(self.model.system.conv1.weight.data - conv1_weight_orig).abs().mean():.4f}")
        print(f"  Conv2权重变化: {(self.model.system.conv2.weight.data - conv2_weight_orig).abs().mean():.4f}")

    def _hebbian_update_conv(
        self,
        images: torch.Tensor,
        conv_layer: nn.Conv2d,
        original_weights: torch.Tensor,
    ):
        """
        对卷积层执行Hebbian更新

        策略：
        - 记录前突触活动（输入特征图）
        - 记录后突触活动（输出特征图）
        - 用外积更新权重
        """
        # 前突触：输入特征
        pre = images

        # 后突触：卷积输出（用于Hebbian更新）
        with torch.no_grad():
            post = torch.nn.functional.relu(
                conv_layer(pre)
            )

        # 计算Hebbian更新
        # 简化：使用批次平均的相关性
        batch_size = pre.shape[0]

        # 更新权重
        weight_delta = torch.zeros_like(conv_layer.weight.data)

        for b in range(batch_size):
            # 外积: [C_out, C_in, H, W] = [C_out, 1, 1, 1] - [1, C_in, H, W]
            # 简化：使用全局平均
            pre_mean = pre[b].mean()
            post_mean = post[b].mean()

            # Δw = η * pre * post
            weight_delta += self.lr * pre_mean * post_mean

        weight_delta = weight_delta / batch_size

        # Oja规则：防止权重爆炸
        oja_term = 0.001 * conv_layer.weight.data * (post_mean ** 2)
        weight_delta = weight_delta - oja_term

        # 更新权重（保持在原始权重附近的小扰动）
        max_perturbation = 0.5
        perturbation = conv_layer.weight.data - original_weights
        perturbation = torch.clamp(
            perturbation + weight_delta,
            -max_perturbation, max_perturbation
        )
        conv_layer.weight.data = original_weights + perturbation

    def _simple_reconstruct(self, features: np.ndarray) -> np.ndarray:
        """简单重建（用于计算损失）"""
        return features * 0.9

    def train_finetune(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
    ):
        """
        BP微调阶段

        固定Hebbian预训练的特征层，只微调分类器
        """
        print("\n" + "=" * 50)
        print("阶段2: BP微调分类器")
        print("=" * 50)

        for epoch in range(self.finetune_epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                if batch_idx % 100 == 0:
                    print(f"  Epoch {epoch+1}/{self.finetune_epochs} "
                          f"[Batch {batch_idx}/{len(train_loader)}] "
                          f"Loss: {loss.item():.4f} "
                          f"Acc: {100.*correct/total:.2f}%")

            avg_loss = total_loss / len(train_loader)
            acc = 100. * correct / total
            self.history['finetune']['loss'].append(avg_loss)
            self.history['finetune']['acc'].append(acc)

            print(f"  Epoch {epoch+1} 完成, Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")

        # 最终测试
        self.evaluate(test_loader)

    def evaluate(self, test_loader: DataLoader):
        """评估模型"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total
        print(f"\n  测试准确率: {acc:.2f}%")
        return acc

    def plot_training_curve(self, save_path: str = None):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Hebbian阶段
        axes[0].plot(self.history['hebbian']['loss'], 'b-', linewidth=2)
        axes[0].set_title('Hebbian预训练损失')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True)

        # Finetune阶段
        axes[1].plot(self.history['finetune']['loss'], 'r-', linewidth=2,
                    label='Loss')
        axes[1].plot(self.history['finetune']['acc'], 'g--', linewidth=2,
                    label='Accuracy')
        axes[1].set_title('BP微调')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Value')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(save_path if save_path else 'training_curve.png')
        plt.show()


# ============================================================
# BP基线模型（对照组）
# ============================================================

class StandardCNN(nn.Module):
    """标准CNN（纯BP训练）"""

    def __init__(self, n_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_bp_baseline(
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 10,
    lr: float = 0.001,
) -> tuple:
    """训练标准BP模型（对照组）"""
    print("\n" + "=" * 50)
    print("BP基线训练（对照组）")
    print("=" * 50)

    model = StandardCNN(n_classes=10)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {'loss': [], 'acc': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        acc = 100. * correct / total
        history['loss'].append(avg_loss)
        history['acc'].append(acc)

        print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")

    # 测试
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = 100. * correct / total
    print(f"\n  测试准确率: {test_acc:.2f}%")

    return model, history, test_acc


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("Hebbian CNN 训练")
    print("=" * 60)

    # 加载数据
    print("\n加载MNIST数据集...")
    train_loader, test_loader = load_mnist(batch_size=128)
    print(f"  训练样本: {len(train_loader.dataset)}")
    print(f"  测试样本: {len(test_loader.dataset)}")

    # Hebbian + BP
    print("\n" + "#" * 60)
    print("# Hebbian预训练 + BP微调")
    print("#" * 60)

    trainer = HebbianTrainer(
        n_classes=10,
        hebbian_epochs=3,
        finetune_epochs=5,
        learning_rate=0.001,
    )

    trainer.train_hebbian(train_loader)
    trainer.train_finetune(train_loader, test_loader)
    trainer.plot_training_curve('hebbian_training.png')

    # BP基线
    print("\n" + "#" * 60)
    print("# BP基线（对照组）")
    print("#" * 60)

    bp_model, bp_history, bp_acc = train_bp_baseline(
        train_loader, test_loader, epochs=5
    )

    # 对比结果
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    print(f"  Hebbian + BP: 训练完成")
    print(f"  纯BP基线: {bp_acc:.2f}%")

    print("\n训练完成！")


if __name__ == '__main__':
    main()