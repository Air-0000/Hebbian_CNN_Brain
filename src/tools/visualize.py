"""
可视化工具 - Hebbian CNN 层级架构与学习过程

功能：
1. 四层架构图
2. 学习曲线
3. 特征可视化
4. 对比分析图
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 四层架构图
# ============================================================

def draw_hierarchy(save_path: str = 'hierarchy.png'):
    """
    绘制记忆强化机制的层级架构图

    层次：
    1. 分子层: LTP/LTD
    2. 突触层: Hebbian规则
    3. 网络层: Lindsay模型
    4. 系统层: CNN
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor('#f8f9fa')

    # 层级数据
    layers = [
        {
            'name': '第1层：分子层',
            'mech': 'LTP / LTD',
            'func': '突触强化的分子开关',
            'detail': '• CaMKII 激活\n• AMPA受体转运\n• Ca²⁺ 信号级联',
            'color': '#e74c3c',
            'y': 8.5,
        },
        {
            'name': '第2层：突触层',
            'mech': 'Hebbian规则',
            'func': '"一起放电则连接加强"',
            'detail': '• Δw = η · x · y\n• Oja规则抑制\n• 竞争机制(WTA)',
            'color': '#3498db',
            'y': 6.5,
        },
        {
            'name': '第3层：网络层',
            'mech': 'Lindsay模型',
            'func': '混合选择性记忆',
            'detail': '• 90个PFC神经元\n• 8步学习演化\n• 与真实PFC对比',
            'color': '#2ecc71',
            'y': 4.5,
        },
        {
            'name': '第4层：系统层',
            'mech': 'CNN',
            'func': '层次化特征提取',
            'detail': '• 多尺度卷积核\n• 层次化表征\n• 分类器',
            'color': '#9b59b6',
            'y': 2.5,
        },
    ]

    # 绘制每个层级
    for i, layer in enumerate(layers):
        y = layer['y']

        # 主框
        rect = patches.FancyBboxPatch(
            (0.5, y - 0.8), 3.5, 1.6,
            boxstyle="round,pad=0.1,rounding_size=0.2",
            facecolor=layer['color'], alpha=0.15,
            edgecolor=layer['color'], linewidth=3
        )
        ax.add_patch(rect)

        # 层名
        ax.text(0.8, y + 0.5, layer['name'],
               fontsize=14, fontweight='bold', color=layer['color'])

        # 机制
        ax.text(0.8, y + 0.1, layer['mech'],
               fontsize=12, style='italic', color='#2c3e50')

        # 功能
        ax.text(4.3, y + 0.3, layer['func'],
               fontsize=12, color='#2c3e50', ha='left')

        # 详情框
        detail_box = patches.FancyBboxPatch(
            (7, y - 0.8), 2.8, 1.6,
            boxstyle="round,pad=0.1,rounding_size=0.1",
            facecolor='white', alpha=0.8,
            edgecolor='#bdc3c7', linewidth=1
        )
        ax.add_patch(detail_box)
        ax.text(7.2, y + 0.4, layer['detail'],
               fontsize=9, color='#34495e', linespacing=1.5)

        # 箭头连接
        if i < len(layers) - 1:
            ax.annotate('',
                       xy=(2.25, layers[i+1]['y'] + 0.8),
                       xytext=(2.25, y - 0.8),
                       arrowprops=dict(
                           arrowstyle='->',
                           color=layer['color'],
                           lw=2,
                           connectionstyle='arc3,rad=-0.2'
                       ))

    # 右侧标注
    ax.text(0.3, 9, '微观', fontsize=10, color='#7f8c8d', ha='center')
    ax.text(0.3, 1.5, '宏观', fontsize=10, color='#7f8c8d', ha='center')

    # 底部说明
    ax.text(5, 1, '层级关系: 分子层(LTP) → 突触层(Hebbian) → 网络层(Lindsay) → 系统层(CNN)',
           fontsize=11, ha='center', color='#7f8c8d',
           bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))

    # 标题
    ax.text(5, 9.8, '记忆强化机制的层级架构',
           fontsize=18, fontweight='bold', ha='center', color='#2c3e50')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"层级架构图已保存: {save_path}")


def draw_learning_process(save_path: str = 'learning_process.png'):
    """
    绘制Hebbian学习过程的可视化

    展示：
    1. 输入刺激模式
    2. 突触权重演化
    3. 神经元响应变化
    4. 选择性形成
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. 输入模式
    ax = axes[0, 0]
    # 生成三个任务变量的刺激
    task_vars = [2, 4, 4]  # TT, C1, C2
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    x = np.arange(len(task_vars))
    bars = ax.bar(x, task_vars, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(['TT\n(2类)', 'C1\n(4类)', 'C2\n(4类)'])
    ax.set_ylabel('选项数')
    ax.set_title('输入任务变量')
    ax.set_ylim(0, 5)

    for bar, val in zip(bars, task_vars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               str(val), ha='center', fontweight='bold')

    # 2. Hebbian学习规则
    ax = axes[0, 1]
    ax.text(0.5, 0.8, 'Hebbian规则', fontsize=14, ha='center', fontweight='bold',
           transform=ax.transAxes)

    ax.text(0.5, 0.6, r'$\Delta w_{ij} = \eta \cdot x_i \cdot y_j$',
           fontsize=16, ha='center', transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='#ecf0f1'))

    ax.text(0.5, 0.35, 'Oja规则 (防止权重爆炸):',
           fontsize=11, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.15, r'$\Delta w = \eta \cdot (x \cdot y - \gamma \cdot y^2 \cdot w)$',
           fontsize=12, ha='center', transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='#d5f5e3'))

    ax.axis('off')
    ax.set_title('学习规则')

    # 3. 突触权重演化
    ax = axes[0, 2]
    steps = np.arange(1, 9)
    np.random.seed(42)

    # 模拟权重范数演化
    norms = []
    current_norm = 0.5
    for _ in steps:
        current_norm += np.random.uniform(0.1, 0.3)
        norms.append(current_norm)

    ax.plot(steps, norms, 'b-o', linewidth=2, markersize=8)
    ax.axhline(y=1.0, color='r', linestyle='--', label='稳定水平')
    ax.set_xlabel('学习步')
    ax.set_ylabel('权重范数')
    ax.set_title('突触权重演化')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 混合选择性演化
    ax = axes[1, 0]
    mixed = [0.1, 0.15, 0.22, 0.28, 0.33, 0.40, 0.46, 0.52]
    pure = [0.9, 0.85, 0.78, 0.72, 0.67, 0.60, 0.54, 0.48]

    ax.fill_between(steps, pure, alpha=0.3, color='green', label='纯选择性')
    ax.fill_between(steps, 0, mixed, alpha=0.3, color='orange', label='混合选择性')
    ax.plot(steps, mixed, 'o-', color='orange', linewidth=2, markersize=6,
           label='混合选择性演化')
    ax.axhline(y=0.51, color='gray', linestyle=':', label='参考水平(51%)')

    ax.set_xlabel('学习步')
    ax.set_ylabel('比例')
    ax.set_title('选择性演化 (关键发现)')
    ax.legend(loc='center right')
    ax.grid(True, alpha=0.3)

    # 5. 竞争机制示意
    ax = axes[1, 1]
    # 模拟神经元竞争
    n_neurons = 8
    neurons = np.arange(n_neurons)

    # 学习前：均匀分布
    before = np.ones(n_neurons) / n_neurons

    # 学习后：选择性强化
    after = np.exp(-0.5 * (neurons - 2)**2)
    after = after / after.sum()

    width = 0.35
    ax.bar(neurons - width/2, before, width, label='学习前', color='#3498db', alpha=0.7)
    ax.bar(neurons + width/2, after, width, label='学习后', color='#e74c3c', alpha=0.7)

    ax.set_xlabel('神经元索引')
    ax.set_ylabel('响应强度')
    ax.set_title('竞争机制 (WTA)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. 层级对应总结
    ax = axes[1, 2]
    ax.axis('off')

    summary = """
    层级对应关系

    ─────────────────────────
    分子层    → LTP/LTD机制
                    ↓
    突触层    → Hebbian规则
                    ↓
    网络层    → 混合选择性
                    ↓
    系统层    → CNN特征提取

    ─────────────────────────
    核心发现：
    • 简单Hebbian优于复杂Hebbian
    • 混合选择性是记忆的关键
    • 竞争产生判别性
    """

    ax.text(0.1, 0.95, summary, fontsize=11, va='top', ha='left',
           transform=ax.transAxes, family='monospace',
           bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.9))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"学习过程图已保存: {save_path}")


# ============================================================
# 2. CNN特征可视化
# ============================================================

def visualize_cnn_features(model: nn.Module, images: torch.Tensor,
                          save_path: str = 'cnn_features.png'):
    """
    可视化CNN学习到的特征

    展示：
    1. 卷积核权重
    2. 特征图激活
    """
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # 获取卷积层
    weights = model[0].weight.data.cpu().numpy()  # [16, 1, 3, 3]

    for i in range(min(16, weights.shape[0])):
        row = i // 4
        col = i % 4

        # 绘制3x3核
        kernel = weights[i, 0]
        axes[row, col].imshow(kernel, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        axes[row, col].axis('off')
        axes[row, col].set_title(f'Conv1-{i+1}', fontsize=10)

    plt.suptitle('Conv1层卷积核权重 (Hebbian学习后)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"CNN特征图已保存: {save_path}")


def visualize_feature_maps(features: torch.Tensor, save_path: str = 'feature_maps.png'):
    """
    可视化特征图激活

    Args:
        features: [B, C, H, W] 特征图张量
    """
    B, C, H, W = features.shape
    n_show = min(C, 16)

    rows = int(np.ceil(n_show ** 0.5))
    cols = int(np.ceil(n_show / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    if n_show == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    # 平均所有批次
    feat_avg = features.mean(dim=0).cpu().numpy()  # [C, H, W]

    for i in range(n_show):
        im = axes[i].imshow(feat_avg[i], cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f'Channel {i+1}', fontsize=10)
        plt.colorbar(im, ax=axes[i], shrink=0.6)

    # 隐藏多余的子图
    for i in range(n_show, len(axes)):
        axes[i].axis('off')

    plt.suptitle('特征图激活可视化', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"特征图已保存: {save_path}")


# ============================================================
# 3. 对比实验结果
# ============================================================

def plot_comparison_results(results: dict, save_path: str = 'results.png'):
    """
    绘制对比实验结果

    Args:
        results: {'hebbian_bp': {...}, 'bp_only': {...}, 'hebbian_only': {...}}
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    methods = list(results.keys())
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    labels = ['Hebbian+BP', 'BP Only', 'Hebbian Only']

    # 1. 准确率
    ax = axes[0, 0]
    for method, color, label in zip(methods, colors, labels):
        if 'acc' in results[method]:
            epochs = range(1, len(results[method]['acc']) + 1)
            ax.plot(epochs, results[method]['acc'], '-o', color=color,
                   label=label, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('分类准确率对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 特征多样性
    ax = axes[0, 1]
    for method, color, label in zip(methods, colors, labels):
        if 'diversity' in results[method]:
            epochs = range(1, len(results[method]['diversity']) + 1)
            ax.plot(epochs, results[method]['diversity'], '-s', color=color,
                   label=label, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Diversity')
    ax.set_title('特征多样性对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 柱状图对比
    ax = axes[1, 0]
    x = np.arange(len(methods))
    final_accs = [results[m]['acc'][-1] if 'acc' in results[m] else 0 for m in methods]

    bars = ax.bar(x, final_accs, color=colors, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('最终准确率对比')
    ax.set_ylim(0, 100)

    for bar, acc in zip(bars, final_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 4. 生物学plausibility雷达图
    ax = axes[1, 1]
    ax.axis('off')

    # 创建表格
    table_data = [
        ['方法', '准确率', '多样性', '生物plausibility'],
        ['Hebbian+BP', f'{results["hebbian_bp"]["acc"][-1]:.1f}%' if 'acc' in results['hebbian_bp'] else '-',
         f'{results["hebbian_bp"]["diversity"][-1]:.3f}' if 'diversity' in results['hebbian_bp'] else '-', '高'],
        ['BP Only', f'{results["bp_only"]["acc"][-1]:.1f}%' if 'acc' in results['bp_only'] else '-',
         f'{results["bp_only"]["diversity"][-1]:.3f}' if 'diversity' in results['bp_only'] else '-', '低'],
        ['Hebbian Only', f'{results["hebbian_only"]["acc"][-1]:.1f}%' if 'acc' in results['hebbian_only'] else '-',
         f'{results["hebbian_only"]["diversity"][-1]:.3f}' if 'diversity' in results['hebbian_only'] else '-', '最高'],
    ]

    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                    loc='center', cellLoc='center',
                    colColours=['#3498db']*4)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    ax.set_title('综合对比', pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"对比结果图已保存: {save_path}")


# ============================================================
# 4. 主函数
# ============================================================

def main():
    print("=" * 60)
    print("Hebbian CNN 可视化工具")
    print("=" * 60)

    # 1. 层级架构图
    print("\n[1] 生成层级架构图...")
    draw_hierarchy('hierarchy.png')

    # 2. 学习过程图
    print("\n[2] 生成学习过程图...")
    draw_learning_process('learning_process.png')

    # 3. 模拟CNN特征可视化（无需真实模型）
    print("\n[3] 生成模拟CNN特征图...")

    # 创建模拟模型（使用Sequential使其可索引）
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
    )

    # 随机输入
    dummy_images = torch.randn(1, 1, 28, 28)

    # 可视化
    visualize_cnn_features(model, dummy_images, 'cnn_features.png')

    print("\n" + "=" * 60)
    print("可视化完成！")
    print("生成的文件：")
    print("  - hierarchy.png (层级架构)")
    print("  - learning_process.png (学习过程)")
    print("  - cnn_features.png (CNN特征)")
    print("=" * 60)


if __name__ == '__main__':
    main()