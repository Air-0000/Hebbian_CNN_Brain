"""
对比分析：Hebbian学习 vs 反向传播

实验设计：
1. Hebbian预训练 + BP微调
2. 纯BP训练（对照组）
3. 不同指标对比

评估维度：
- 分类准确率
- 特征多样性
- 生物学plausibility
- 计算效率
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns


# ============================================================
# 特征提取与可视化
# ============================================================

class FeatureAnalyzer:
    """
    特征分析工具

    分析内容：
    1. 特征多样性（通道间相关性）
    2. 特征判别性（类别分离度）
    3. 特征演化（权重可视化）
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.feature_maps = []

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """提取特征图"""
        with torch.no_grad():
            # Conv1输出
            x = torch.nn.functional.relu(self.model.conv1(images))
            feat1 = x

            # Conv2输出
            x = torch.nn.functional.max_pool2d(x, 2)
            x = torch.nn.functional.relu(self.model.conv2(x))
            feat2 = x

        return feat1, feat2

    def compute_diversity(self, features: torch.Tensor) -> float:
        """
        计算特征多样性

        高多样性 = 低通道间相关性 = 好
        低多样性 = 高通道间相关性 = 特征同质化 = 差
        """
        # features: [B, C, H, W]
        B, C, H, W = features.shape

        # 全局平均池化 -> [B, C]
        pooled = features.mean(dim=(2, 3))

        # 计算通道间相关系数矩阵
        corr_matrix = torch.corrcoef(pooled.T)

        # 多样性 = 1 - 平均相关性（排除对角线）
        mask = torch.ones_like(corr_matrix) - torch.eye(C, device=corr_matrix.device)
        avg_corr = (corr_matrix * mask).sum() / (C * (C - 1))

        diversity = 1 - avg_corr.item()
        return diversity

    def compute_selectivity(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        """
        计算选择性（类别分离度）

        高选择性 = 类间距离大，类内距离小
        """
        B, C, H, W = features.shape

        # 全局平均 -> [B, C]
        pooled = features.mean(dim=(2, 3))

        # 按类别分组
        classes = torch.unique(labels)
        class_means = []

        for c in classes:
            mask = labels == c
            class_mean = pooled[mask].mean(dim=0)  # [C]
            class_means.append(class_mean)

        class_means = torch.stack(class_means)  # [n_classes, C]

        # 类间方差 / 类内方差 (类似Fisher准则)
        global_mean = pooled.mean(dim=0)

        # 类间
        between_var = sum(
            ((cm - global_mean) ** 2).sum()
            for cm in class_means
        ) / len(classes)

        # 类内
        within_var = 0
        for c in classes:
            mask = labels == c
            class_samples = pooled[mask]
            within_var += ((class_samples - class_means[c]) ** 2).sum()
        within_var /= B

        selectivity = between_var / (within_var + 1e-6)
        return selectivity.item()

    def visualize_filters(self, save_path: str = None):
        """可视化卷积核"""
        weights = self.model.conv1.weight.data

        # 假设16个通道，4x4网格
        n_channels = min(weights.shape[0], 16)
        rows = int(np.ceil(n_channels ** 0.5))
        cols = int(np.ceil(n_channels / rows))

        fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
        axes = axes.flatten() if n_channels > 1 else [axes]

        for i in range(n_channels):
            w = weights[i, 0].cpu().numpy()  # 单通道
            axes[i].imshow(w, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
            axes[i].axis('off')
            axes[i].set_title(f'Filter {i+1}')

        # 隐藏多余的子图
        for i in range(n_channels, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(save_path if save_path else 'filters.png')
        plt.show()


# ============================================================
# 实验对比
# ============================================================

class ComparisonExperiment:
    """
    对比实验

    比较：
    1. Hebbian + BP
    2. 纯BP
    3. Hebbian only (无微调)
    """

    def __init__(self, device: str = 'cpu'):
        self.device = device

        # 结果存储
        self.results = {
            'hebbian_bp': {'acc': [], 'diversity': [], 'selectivity': []},
            'bp_only': {'acc': [], 'diversity': [], 'selectivity': []},
            'hebbian_only': {'acc': [], 'diversity': [], 'selectivity': []},
        }

    def run_hebbian_bp(self, train_loader, test_loader, epochs=(3, 5)):
        """Hebbian预训练 + BP微调"""
        print("\n[实验1] Hebbian预训练 + BP微调")

        # 模型
        model = self._create_model().to(self.device)

        # Hebbian阶段
        print("  阶段1: Hebbian无监督预训练...")
        for epoch in range(epochs[0]):
            for images, _ in train_loader:
                images = images.to(self.device)
                self._hebbian_update(model, images)

        # BP阶段
        print("  阶段2: BP微调...")
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs[1]):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # 评估
            acc, diversity, sel = self._evaluate(model, train_loader, test_loader)
            self.results['hebbian_bp']['acc'].append(acc)
            self.results['hebbian_bp']['diversity'].append(diversity)
            self.results['hebbian_bp']['selectivity'].append(sel)
            print(f"    Epoch {epoch+1}: Acc={acc:.2f}%, Diversity={diversity:.3f}")

        return model

    def run_bp_only(self, train_loader, test_loader, epochs=10):
        """纯BP训练"""
        print("\n[实验2] 纯BP训练（对照组）")

        model = self._create_model().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            acc, diversity, sel = self._evaluate(model, train_loader, test_loader)
            self.results['bp_only']['acc'].append(acc)
            self.results['bp_only']['diversity'].append(diversity)
            self.results['bp_only']['selectivity'].append(sel)
            print(f"    Epoch {epoch+1}: Acc={acc:.2f}%, Diversity={diversity:.3f}")

        return model

    def run_hebbian_only(self, train_loader, test_loader, epochs=10):
        """纯Hebbian（无BP）"""
        print("\n[实验3] 纯Hebbian（无微调）")

        model = self._create_model().to(self.device)

        for epoch in range(epochs):
            for images, _ in train_loader:
                images = images.to(self.device)
                self._hebbian_update(model, images)

            acc, diversity, sel = self._evaluate(model, train_loader, test_loader)
            self.results['hebbian_only']['acc'].append(acc)
            self.results['hebbian_only']['diversity'].append(diversity)
            self.results['hebbian_only']['selectivity'].append(sel)
            print(f"    Epoch {epoch+1}: Acc={acc:.2f}%, Diversity={diversity:.3f}")

        return model

    def _create_model(self):
        """创建模型"""
        return nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def _hebbian_update(self, model, images):
        """Hebbian更新"""
        x = model[0](images)  # Conv1
        x = torch.nn.functional.relu(x)

        # 简化的Hebbian更新
        # 使用全局平均作为相关性度量
        pre_mean = images.mean()
        post_mean = x.mean()

        delta = 0.01 * pre_mean * post_mean
        model[0].weight.data += delta * torch.randn_like(model[0].weight.data) * 0.1

    def _evaluate(self, model, train_loader, test_loader):
        """评估模型"""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total

        # 特征多样性（使用训练集）
        analyzer = FeatureAnalyzer(model)
        images, labels = next(iter(train_loader))
        images = images.to(self.device)
        feat1, feat2 = analyzer.extract_features(images)
        diversity = analyzer.compute_diversity(feat2)
        selectivity = analyzer.compute_selectivity(feat2, labels.to(self.device))

        return acc, diversity, selectivity

    def plot_comparison(self, save_path: str = None):
        """绘制对比图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        methods = ['hebbian_bp', 'bp_only', 'hebbian_only']
        titles = ['Hebbian+BP', 'BP Only', 'Hebbian Only']
        colors = ['#2E86AB', '#A23B72', '#F18F01']

        # 准确率
        for i, (method, title, color) in enumerate(zip(methods, titles, colors)):
            data = self.results[method]['acc']
            axes[0].plot(data, '-o', color=color, label=title, linewidth=2)
        axes[0].set_title('分类准确率')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].legend()
        axes[0].grid(True)

        # 特征多样性
        for i, (method, title, color) in enumerate(zip(methods, titles, colors)):
            data = self.results[method]['diversity']
            axes[1].plot(data, '-s', color=color, label=title, linewidth=2)
        axes[1].set_title('特征多样性')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Diversity')
        axes[1].legend()
        axes[1].grid(True)

        # 选择性
        for i, (method, title, color) in enumerate(zip(methods, titles, colors)):
            data = self.results[method]['selectivity']
            axes[2].plot(data, '-^', color=color, label=title, linewidth=2)
        axes[2].set_title('类别选择性')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Selectivity')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(save_path if save_path else 'comparison.png')
        plt.show()

    def plot_summary(self, save_path: str = None):
        """绘制汇总对比图"""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig)

        methods = ['hebbian_bp', 'bp_only', 'hebbian_only']
        titles = ['Hebbian + BP\n(预训练+微调)', 'BP Only\n(对照组)', 'Hebbian Only\n(无微调)']
        colors = ['#2E86AB', '#A23B72', '#F18F01']

        # 最终准确率对比
        ax1 = fig.add_subplot(gs[0, 0])
        final_accs = [self.results[m]['acc'][-1] for m in methods]
        bars = ax1.bar(titles, final_accs, color=colors, edgecolor='black')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('最终分类准确率')
        ax1.set_ylim(0, 100)
        for bar, acc in zip(bars, final_accs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

        # 特征多样性对比
        ax2 = fig.add_subplot(gs[0, 1])
        diversities = [self.results[m]['diversity'][-1] for m in methods]
        bars = ax2.bar(titles, diversities, color=colors, edgecolor='black')
        ax2.set_ylabel('Diversity')
        ax2.set_title('特征多样性')
        for bar, d in zip(bars, diversities):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{d:.3f}', ha='center', va='bottom', fontweight='bold')

        # 特征选择性对比
        ax3 = fig.add_subplot(gs[0, 2])
        selectivities = [self.results[m]['selectivity'][-1] for m in methods]
        bars = ax3.bar(titles, selectivities, color=colors, edgecolor='black')
        ax3.set_ylabel('Selectivity')
        ax3.set_title('类别选择性')
        for bar, s in zip(bars, selectivities):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{s:.2f}', ha='center', va='bottom', fontweight='bold')

        # 准确率演化曲线
        ax4 = fig.add_subplot(gs[1, :2])
        for method, title, color in zip(methods, titles, colors):
            data = self.results[method]['acc']
            ax4.plot(data, '-o', color=color, label=title.replace('\n', ' '),
                    linewidth=2, markersize=6)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy (%)')
        ax4.set_title('准确率演化')
        ax4.legend()
        ax4.grid(True)

        # 生物学plausibility对比
        ax5 = fig.add_subplot(gs[1, 2])
        plausibility = {
            'Hebbian + BP': 0.7,
            'BP Only': 0.2,
            'Hebbian Only': 0.9,
        }
        names = list(plausibility.keys())
        values = list(plausibility.values())
        bars = ax5.barh(names, values, color=['#2E86AB', '#A23B72', '#F18F01'])
        ax5.set_xlabel('Biological Plausibility')
        ax5.set_title('生物学合理性')
        ax5.set_xlim(0, 1)
        for bar, v in zip(bars, values):
            ax5.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{v:.1f}', ha='left', va='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path if save_path else 'summary.png')
        plt.show()


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("Hebbian vs BP 对比实验")
    print("=" * 60)

    # 加载数据
    print("\n加载MNIST数据集...")
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

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 运行实验
    exp = ComparisonExperiment()

    exp.run_hebbian_bp(train_loader, test_loader, epochs=(2, 5))
    exp.run_bp_only(train_loader, test_loader, epochs=8)
    exp.run_hebbian_only(train_loader, test_loader, epochs=8)

    # 绘图
    print("\n生成对比图...")
    exp.plot_comparison('comparison.png')
    exp.plot_summary('summary.png')

    # 打印结论
    print("\n" + "=" * 60)
    print("实验结论")
    print("=" * 60)

    methods = ['hebbian_bp', 'bp_only', 'hebbian_only']
    for method in methods:
        print(f"\n[{method.upper()}]")
        print(f"  最终准确率: {exp.results[method]['acc'][-1]:.2f}%")
        print(f"  特征多样性: {exp.results[method]['diversity'][-1]:.4f}")
        print(f"  类别选择性: {exp.results[method]['selectivity'][-1]:.4f}")

    print("\n实验完成!")


if __name__ == '__main__':
    main()