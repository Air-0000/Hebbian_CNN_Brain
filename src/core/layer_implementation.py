"""
记忆强化机制的层级实现

四个层次：
1. 分子层 (Molecular): LTP/LTD机制模拟
2. 突触层 (Synaptic): Hebbian规则
3. 网络层 (Network): Lindsay前额叶模型 + 混合选择性
4. 系统层 (System): CNN层次化特征提取

层级关系:
分子层(LTP) → 突触层(Hebbian) → 网络层(Lindsay) → 系统层(CNN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt


# ============================================================
# 第1层：分子层 - LTP/LTD机制
# ============================================================

class LTPMechanism:
    """
    长时程增强(Long-Term Potentiation) 分子机制

    模拟步骤：
    1. CaMKII激活（钙调蛋白激酶II）
    2. AMPAR磷酸化
    3. 受体转运到突触膜
    4. 突触强度增强

    这是记忆强化的分子基础
    """

    def __init__(self, tau_ca: float = 0.1, tau_amp: float = 0.5):
        # CaMKII参数
        self.tau_ca = tau_ca  # 钙离子信号衰减时间常数
        self.tau_amp = tau_amp  # AMPAR插入/移除时间常数

        # 初始状态
        self.camkii_activity = 0.0  # CaMKII激活水平 [0,1]
        self.ampar_density = 1.0   # AMPA受体密度 (相对值)
        self.ltp_strength = 0.0    # LTP增强强度 [0,1]

    def step(self, pre_spike: float, post_spike: float, dt: float = 1.0) -> Dict:
        """
        LTP触发步骤

        Args:
            pre_spike: 前突触发放率 [0,1]
            post_spike: 后突触发放率 [0,1]
            dt: 时间步长

        Returns:
            更新后的状态字典
        """
        # 计算相关性（Hebbian的核心）
        correlation = pre_spike * post_spike

        # 1. 钙离子内流
        ca_influx = correlation * 0.8  # 高相关性 → 大量Ca²⁺

        # 2. CaMKII激活
        # CaMKII是分子开关，需要足够的Ca²⁺才能激活
        camkii_input = ca_influx
        self.camkii_activity += dt * (
            -self.camkii_activity / self.tau_ca + camkii_input
        )
        self.camkii_activity = np.clip(self.camkii_activity, 0, 1)

        # 3. AMPA受体转运
        # CaMKII磷酸化 → 促进AMPAR插入突触膜
        if self.camkii_activity > 0.3:  # 阈值
            # AMPAR插入（增强）
            self.ampar_density += 0.05 * self.camkii_activity * (1 - self.ampar_density)
        else:
            # AMPAR移除（基础水平）
            self.ampar_density -= 0.01 * (self.ampar_density - 1.0)

        self.ampar_density = np.clip(self.ampar_density, 0.5, 3.0)

        # 4. LTP增强强度计算
        self.ltp_strength = self.camkii_activity * (self.ampar_density - 1.0)

        return {
            'ca_influx': ca_influx,
            'camkii_activity': self.camkii_activity,
            'ampar_density': self.ampar_density,
            'ltp_strength': self.ltp_strength,
            'correlation': correlation,
        }

    def ltd_trigger(self, pre_spike: float, post_spike: float) -> float:
        """
        LTD触发（低频刺激）

        轻度Ca²⁺内流 → 激活磷酸酶 → 移除AMPAR
        """
        correlation = pre_spike * post_spike

        if correlation < 0.1:  # 低相关性
            ltd_strength = -0.02 * correlation
            self.ampar_density += ltd_strength
            self.ampar_density = np.clip(self.ampar_density, 0.5, 3.0)
            return ltd_strength
        return 0.0


class MolecularLayer:
    """
    分子层：LTP/LTD分子机制的向量化实现

    适用于大规模突触阵列
    """

    def __init__(self, n_synapses: int):
        self.n_synapses = n_synapses

        # 每个突触的分子状态
        self.camkii = np.zeros(n_synapses)  # CaMKII激活
        self.ampar = np.ones(n_synapses)     # AMPA受体密度
        self.ltp = np.zeros(n_synapses)      # LTP强度

    def forward(self, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
        """
        分子层前向传播

        Args:
            pre: 前突触活动 [n_synapses]
            post: 后突触活动 [n_synapses]

        Returns:
            LTP增强量 [n_synapses]
        """
        # 相关性
        corr = pre * post

        # CaMKII激活（阈值机制）
        self.camkii = 0.9 * self.camkii + 0.1 * (corr > 0.2) * corr

        # AMPA受体更新
        self.ampar += 0.01 * self.camkii * (1 - self.ampar)
        self.ampar -= 0.005 * (corr < 0.1) * (self.ampar - 1.0)
        self.ampar = np.clip(self.ampar, 0.5, 3.0)

        # LTP计算
        self.ltp = self.camkii * (self.ampar - 1.0)

        return self.ltp

    def get_state(self) -> Dict:
        """获取分子层状态"""
        return {
            'camkii_mean': self.camkii.mean(),
            'ampar_mean': self.ampar.mean(),
            'ltp_mean': self.ltp.mean(),
        }


# ============================================================
# 第2层：突触层 - Hebbian规则
# ============================================================

class SynapticLayer:
    """
    突触层：Hebbian学习规则

    核心公式：
    - 基本Hebbian: Δw = η * x * y
    - Oja规则: Δw = η * (x * y - γ * y² * w)

    整合分子层的LTP信号
    """

    def __init__(self, n_pre: int, n_post: int, lr: float = 0.01):
        self.n_pre = n_pre
        self.n_post = n_post
        self.lr = lr

        # 权重矩阵 [n_post, n_pre]
        self.weights = np.random.randn(n_post, n_pre) * 0.01

        # Oja抑制系数
        self.gamma = 0.001

        # 可选：分子层接口
        self.molecular = None

    def set_molecular_layer(self, molecular: MolecularLayer):
        """连接分子层"""
        self.molecular = molecular

    def forward(self, pre: np.ndarray) -> np.ndarray:
        """
        突触前向传播

        y = W @ x
        """
        return self.weights @ pre

    def hebbian_update(self, pre: np.ndarray, post: np.ndarray,
                       use_molecular: bool = True):
        """
        Hebbian权重更新

        Args:
            pre: 前突触活动 [n_pre]
            post: 后突触活动 [n_post]
            use_molecular: 是否使用分子层LTP信号
        """
        # 基本Hebbian更新
        delta_w = self.lr * np.outer(post, pre)

        # Oja规则抑制
        if self.gamma > 0:
            oja_term = self.gamma * np.outer(post**2, np.ones(self.n_pre))
            delta_w -= oja_term * self.weights

        # 整合分子层LTP信号
        if use_molecular and self.molecular is not None:
            ltp_signal = self.molecular.forward(pre, post)
            # LTP增强Hebbian更新
            ltp_factor = 1.0 + ltp_signal.mean() * 0.5
            delta_w *= ltp_factor

        # 更新权重
        self.weights += delta_w

        # 权重正则化（防止数值爆炸）
        self.weights = np.clip(self.weights, -2, 2)

    def get_weight_norm(self) -> np.ndarray:
        """每列的权重范数"""
        return np.sqrt((self.weights ** 2).sum(axis=1))


# ============================================================
# 第3层：网络层 - Lindsay模型 + 混合选择性
# ============================================================

class NetworkNeuron:
    """
    神经元模型

    动力学方程:
    r_i = σ(Σ w_ij * x_j - θ_i)

    整合Lindsay模型的特性:
    - 混合选择性
    - 竞争机制
    """

    def __init__(self, n_inputs: int):
        self.n_inputs = n_inputs

        # 突触权重
        self.weights = np.random.randn(n_inputs) * 0.01

        # 阈值
        self.threshold = 0.0

        # 混合选择性特征
        self.preference_vector = None  # 偏好方向
        self.selectivity = 0.0          # 选择性强度

    def forward(self, inputs: np.ndarray) -> float:
        """计算神经元的发放率"""
        # 加权和
        weighted_sum = np.dot(self.weights, inputs) - self.threshold

        # ReLU激活
        return max(0, weighted_sum)

    def update_selectivity(self, inputs: np.ndarray, responses: np.ndarray):
        """更新选择性特征（用于分析）"""
        # 简化的选择性计算
        if np.std(inputs) > 0:
            correlation = np.corrcoef(inputs, responses)[0, 1]
            self.selectivity = abs(correlation)


class LindsayNetwork:
    """
    Lindsay前额叶模型实现

    关键特性：
    1. 随机网络初始化
    2. Hebbian学习产生混合选择性
    3. 竞争机制（WTA）
    4. 与真实PFC数据对比

    实验参数（Lindsay 2017）：
    - 基底神经元数/群体: 50个
    - 连接概率: 0.25
    - 学习步数: 8步
    - 学习率: 0.2
    - 阈值: 0.27
    """

    def __init__(
        self,
        task_variables: List[int] = [2, 4, 4],  # TT, C1, C2的选项数
        baseline_per_pop: int = 50,
        connection_prob: float = 0.25,
        learning_rate: float = 0.2,
        threshold: float = 0.27,
    ):
        # 任务结构
        self.task_vars = task_variables
        self.n_inputs = np.sum(task_variables) * baseline_per_pop

        # 网络参数
        self.n_neurons = baseline_per_pop * 10  # 90个PFC神经元（与Lindsay一致）
        self.connection_prob = connection_prob
        self.lr = learning_rate
        self.threshold = threshold

        # 创建神经元
        self.neurons = [
            NetworkNeuron(self.n_inputs) for _ in range(self.n_neurons)
        ]

        # 稀疏连接矩阵
        self.connectivity = np.random(
            self.n_neurons, self.n_inputs
        ) < connection_prob

        # 初始化权重（只对有连接的）
        for i, neuron in enumerate(self.neurons):
            connected_inputs = np.where(self.connectivity[i])[0]
            neuron.weights[connected_inputs] = np.random.randn(len(connected_inputs)) * 0.2
            neuron.weights[~self.connectivity[i]] = 0.0

        # 选择性统计
        self.mixed_selectivity = []  # 混合选择性比例（随学习演化）
        self.response_variability = []  # 响应变化度

    def generate_cue_matrix(self, remove_doubles: bool = True) -> np.ndarray:
        """生成任务条件矩阵"""
        n_conditions = np.prod(self.task_vars)
        n_vars = len(self.task_vars)

        cue_mat = np.zeros((n_conditions, n_vars), dtype=int)

        for vi, n_opts in enumerate(self.task_vars):
            reps = np.prod(self.task_vars[vi+1:]) if vi < n_vars-1 else 1
            n_reps = n_conditions // (n_opts * reps)

            for opt in range(n_opts):
                start = opt * reps * n_reps
                cue_mat[start:start + reps * n_reps, vi] = opt

        if remove_doubles:
            # 移除C1=C2的情况
            mask = cue_mat[:, -1] != cue_mat[:, -2]
            cue_mat = cue_mat[mask]

        return cue_mat

    def run_network(self, inputs: np.ndarray, track: bool = True) -> np.ndarray:
        """
        运行网络动力学

        Args:
            inputs: 输入模式 [n_inputs]
            track: 是否追踪神经元活动

        Returns:
            responses: 神经元响应 [n_neurons]
        """
        responses = np.zeros(self.n_neurons)

        for i, neuron in enumerate(self.neurons):
            if self.connectivity[i].any():
                responses[i] = neuron.forward(inputs)
            else:
                responses[i] = 0.0

        return responses

    def hebbian_step(self, cue_matrix: np.ndarray, freelearn: bool = True):
        """
        一步Hebbian学习

        Args:
            cue_matrix: 条件矩阵
            freelearn: True=Free Learning（简单）, False=Constrained（复杂）
        """
        for i, neuron in enumerate(self.neurons):
            connected = np.where(self.connectivity[i])[0]

            if len(connected) == 0:
                continue

            # 遍历每个条件
            for cue in cue_matrix:
                # 生成输入（简化：one-hot编码）
                inputs = np.zeros(self.n_inputs)
                start = 0
                for vi, n_opts in enumerate(self.task_vars):
                    opt = cue[vi]
                    inputs[start + opt * 50 : start + (opt + 1) * 50] = 1.0
                    start += n_opts * 50

                # 稀疏输入
                inputs = inputs[connected] * neuron.weights[connected]
                response = neuron.forward(inputs)

                # Hebbian更新
                if freelearn:
                    # Free Learning: 强化所有活跃连接
                    delta = self.lr * inputs * response
                else:
                    # Constrained: 只强化最强连接的类别
                    # 取前Nl个最强的输入群体
                   Nl = 3
                    class_max = []
                    start = 0
                    for vi, n_opts in enumerate(self.task_vars):
                        class_responses = inputs[start:start + n_opts * 50]
                        class_max.append(np.argmax(class_responses) + start)
                        start += n_opts * 50

                    # 只强化这Nl个类别的连接
                    keep_mask = np.zeros(len(inputs), dtype=bool)
                    for idx in sorted(class_max)[:Nl]:
                        keep_mask[idx * 50:(idx + 1) * 50] = True

                    delta = self.lr * inputs * response * keep_mask

                neuron.weights[connected] += delta

            # 权重归一化
            if neuron.weights.sum() > 0:
                neuron.weights /= neuron.weights.sum()

    def compute_selectivity(self, cue_matrix: np.ndarray) -> Dict:
        """
        计算选择性指标

        与Lindsay论文的ANOVA方法对应
        """
        n_neurons = len(self.neurons)
        n_conditions = len(cue_matrix)

        # 简化的选择性计算
        responses = np.zeros((n_conditions, n_neurons))

        for ci, cue in enumerate(cue_matrix):
            # 构建输入
            inputs = np.zeros(self.n_inputs)
            start = 0
            for vi, n_opts in enumerate(self.task_vars):
                opt = cue[vi]
                inputs[start + opt * 50 : start + (opt + 1) * 50] = 1.0
                start += n_opts * 50

            # 前向传播
            responses[ci] = self.run_network(inputs, track=False)

        # 计算混合选择性（简化版）
        # 如果神经元对多个变量有响应，则是混合选择性
        mixed_count = 0
        pure_count = 0

        for i in range(n_neurons):
            # 计算对每个变量的选择性
            var_selectivity = []
            start = 0
            for vi, n_opts in enumerate(self.task_vars):
                var_responses = []
                for opt in range(n_opts):
                    mask = cue_matrix[:, vi] == opt
                    var_responses.append(responses[mask, i].mean())
                var_selectivity.append(np.std(var_responses))

            # 混合选择性：多个变量都有响应
            if sum(1 for v in var_selectivity if v > 0.1) >= 2:
                mixed_count += 1
            else:
                pure_count += 1

        mixed_ratio = mixed_count / n_neurons
        pure_ratio = pure_count / n_neurons

        return {
            'mixed_ratio': mixed_ratio,
            'pure_ratio': pure_ratio,
            'responses': responses,
        }

    def run_learning(self, n_steps: int = 8, freelearn: bool = True) -> Dict:
        """
        运行完整的学习过程

        模拟Lindsay的实验流程
        """
        cue_matrix = self.generate_cue_matrix()

        results = {
            'step': [],
            'mixed_selectivity': [],
            'weight_norm': [],
            'response_variability': [],
        }

        for step in range(n_steps):
            print(f"学习步 {step + 1}/{n_steps}")

            # Hebbian学习
            self.hebbian_step(cue_matrix, freelearn=freelearn)

            # 计算选择性
            selectivity = self.compute_selectivity(cue_matrix)

            # 计算权重范数
            norms = [np.linalg.norm(n.weights) for n in self.neurons]

            results['step'].append(step + 1)
            results['mixed_selectivity'].append(selectivity['mixed_ratio'])
            results['weight_norm'].append(np.mean(norms))
            results['response_variability'].append(np.std(
                selectivity['responses'].mean(axis=1)
            ))

        return results


# ============================================================
# 第4层：系统层 - CNN层次化特征提取
# ============================================================

class CNNSystemLayer(nn.Module):
    """
    系统层：卷积神经网络

    与下层的关系：
    - Hebbian预训练产生初始化权重
    - 层次化特征提取 = 网络层的放大版本
    - 混合选择性的工程实现：多尺度感受野
    """

    def __init__(
        self,
        n_classes: int = 10,
        hebbian_pretrained_weights: List = None,
    ):
        super().__init__()

        # 卷积层（可加载Hebbian预训练权重）
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # 池化
        self.pool = nn.MaxPool2d(2, 2)

        # 分类器
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, n_classes)

        # 加载Hebbian预训练权重（可选）
        if hebbian_pretrained_weights is not None:
            self.load_hebbian_weights(hebbian_pretrained_weights)

    def load_hebbian_weights(self, weights: List[np.ndarray]):
        """加载Hebbian层预训练的权重"""
        if len(weights) >= 2:
            self.conv1.weight.data = torch.from_numpy(
                weights[0].reshape(16, 1, 3, 3)
            ).float()
            self.conv2.weight.data = torch.from_numpy(
                weights[1].reshape(32, 16, 3, 3)
            ).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class HebbianCNN(nn.Module):
    """
    完整的Hebbian-CNN系统

    整合四个层次：
    1. Molecular: LTP/LTD
    2. Synaptic: Hebbian
    3. Network: Lindsay + Mixed Selectivity
    4. System: CNN Feature Extraction
    """

    def __init__(self, n_classes: int = 10):
        super().__init__()

        # 分子层
        self.molecular = MolecularLayer(n_synapses=1000)

        # 突触层
        self.synaptic = SynapticLayer(n_pre=1000, n_post=500, lr=0.01)
        self.synaptic.set_molecular_layer(self.molecular)

        # 网络层
        self.network = LindsayNetwork(
            task_variables=[2, 4, 4],
            baseline_per_pop=50,
            learning_rate=0.2,
        )

        # 系统层
        self.system = CNNSystemLayer(n_classes=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """完整的前向传播（四个层次的整合）"""
        return self.system(x)

    def get_layer_info(self) -> Dict:
        """获取各层状态信息"""
        return {
            'molecular': self.molecular.get_state(),
            'synaptic_norm': self.synaptic.get_weight_norm().mean(),
            'network_selectivity': {
                'mixed': np.mean(self.network.mixed_selectivity[-5:]) if self.network.mixed_selectivity else 0,
            },
        }


# ============================================================
# 可视化工具
# ============================================================

class LayerVisualizer:
    """四层架构可视化工具"""

    def __init__(self):
        self.records = {
            'molecular': [],
            'synaptic': [],
            'network': [],
            'system': [],
        }

    def update(self, model: HebbianCNN, step: int):
        """记录当前状态"""
        info = model.get_layer_info()

        self.records['molecular'].append({
            'step': step,
            'ltp': info['molecular']['ltp_mean'],
            'camkii': info['molecular']['camkii_mean'],
        })
        self.records['synaptic'].append({
            'step': step,
            'norm': info['synaptic_norm'],
        })
        self.records['network'].append(info['network_selectivity']['mixed'])

    def plot_all(self, save_path: str = None):
        """绘制所有层的演化"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 分子层
        mol_data = self.records['molecular']
        if mol_data:
            steps = [d['step'] for d in mol_data]
            ltp = [d['ltp'] for d in mol_data]
            camkii = [d['camkii'] for d in mol_data]

            axes[0, 0].plot(steps, ltp, 'b-', label='LTP强度', linewidth=2)
            axes[0, 0].plot(steps, camkii, 'r--', label='CaMKII', linewidth=2)
            axes[0, 0].set_title('第1层: 分子层 (LTP/LTD)')
            axes[0, 0].set_xlabel('学习步')
            axes[0, 0].set_ylabel('强度')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

        # 突触层
        syn_data = self.records['synaptic']
        if syn_data:
            steps = [d['step'] for d in syn_data]
            norms = [d['norm'] for d in syn_data]

            axes[0, 1].plot(steps, norms, 'g-', linewidth=2)
            axes[0, 1].set_title('第2层: 突触层 (Hebbian权重)')
            axes[0, 1].set_xlabel('学习步')
            axes[0, 1].set_ylabel('权重范数')
            axes[0, 1].grid(True)

        # 网络层
        net_data = self.records['network']
        if net_data:
            axes[1, 0].plot(range(1, len(net_data) + 1), net_data, 'm-', linewidth=2)
            axes[1, 0].axhline(y=0.51, color='k', linestyle='--', label='参考水平')
            axes[1, 0].set_title('第3层: 网络层 (混合选择性)')
            axes[1, 0].set_xlabel('学习步')
            axes[1, 0].set_ylabel('混合选择性比例')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

        # 系统层
        axes[1, 1].text(0.5, 0.5, '系统层: CNN\n特征提取',
                       ha='center', va='center', fontsize=16)
        axes[1, 1].set_title('第4层: 系统层 (CNN)')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(save_path if save_path else 'layer_evolution.png')
        plt.show()

    def plot_hierarchy(self, save_path: str = None):
        """绘制层级架构图"""
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('off')

        # 层级定义
        layers = [
            {'name': '分子层', 'mech': 'LTP/LTD', 'func': '突触强化分子开关',
             'color': '#FF6B6B', 'y': 0.9},
            {'name': '突触层', 'mech': 'Hebbian规则', 'func': '"一起放电则连接加强"',
             'color': '#4ECDC4', 'y': 0.65},
            {'name': '网络层', 'mech': 'Lindsay模型', 'func': '混合选择性记忆',
             'color': '#45B7D1', 'y': 0.4},
            {'name': '系统层', 'mech': 'CNN', 'func': '层次化特征提取',
             'color': '#96CEB4', 'y': 0.15},
        ]

        for i, layer in enumerate(layers):
            # 层级框
            rect = plt.Rectangle((0.1, layer['y']), 0.8, 0.2,
                                 facecolor=layer['color'], alpha=0.3,
                                 edgecolor=layer['color'], linewidth=3)
            ax.add_patch(rect)

            # 层级名称
            ax.text(0.15, layer['y'] + 0.1, layer['name'],
                   fontsize=18, fontweight='bold', va='center')

            # 机制
            ax.text(0.4, layer['y'] + 0.1, layer['mech'],
                   fontsize=14, va='center', style='italic')

            # 功能
            ax.text(0.7, layer['y'] + 0.1, layer['func'],
                   fontsize=12, va='center')

            # 箭头
            if i < len(layers) - 1:
                ax.annotate('', xy=(0.5, layers[i+1]['y'] + 0.15),
                           xytext=(0.5, layer['y'] - 0.02),
                           arrowprops=dict(arrowstyle='->', color='gray',
                                          lw=2, connectionstyle='arc3,rad=0'))

        # 右侧标注
        ax.text(0.95, 0.9, '微观', fontsize=12, ha='right', va='center')
        ax.text(0.95, 0.15, '宏观', fontsize=12, ha='right', va='center')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('记忆强化机制的四层架构', fontsize=20, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(save_path if save_path else 'hierarchy.png', dpi=150)
        plt.show()


# ============================================================
# 测试
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("记忆强化机制四层实现测试")
    print("=" * 60)

    # 第1层：分子层测试
    print("\n[1] 分子层 - LTP机制")
    mol = MolecularLayer(n_synapses=100)
    for t in range(10):
        result = mol.forward(
            pre=np.random.rand(100) * 0.5,
            post=np.random.rand(100) * 0.5
        )
    state = mol.get_state()
    print(f"  CaMKII: {state['camkii_mean']:.3f}")
    print(f"  AMPA受体: {state['ampar_mean']:.3f}")
    print(f"  LTP强度: {state['ltp_mean']:.3f}")

    # 第2层：突触层测试
    print("\n[2] 突触层 - Hebbian规则")
    syn = SynapticLayer(n_pre=100, n_post=50, lr=0.01)
    syn.set_molecular_layer(mol)
    pre = np.random.rand(100)
    post = syn.forward(pre)
    print(f"  输出维度: {post.shape}")
    print(f"  权重范数: {syn.get_weight_norm().mean():.4f}")

    # 第3层：网络层测试
    print("\n[3] 网络层 - Lindsay模型")
    net = LindsayNetwork(
        task_variables=[2, 4, 4],
        baseline_per_pop=50,
        learning_rate=0.2,
    )
    print(f"  神经元数: {net.n_neurons}")
    print(f"  输入维度: {net.n_inputs}")

    results = net.run_learning(n_steps=5, freelearn=True)
    print(f"  学习完成")
    print(f"  混合选择性: {results['mixed_selectivity'][-1]:.3f}")

    # 第4层：系统层测试
    print("\n[4] 系统层 - CNN")
    model = HebbianCNN(n_classes=10)
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(f"  输入: {x.shape}")
    print(f"  输出: {y.shape}")

    # 可视化
    print("\n[可视化] 生成层级架构图...")
    viz = LayerVisualizer()
    viz.plot_hierarchy('test_hierarchy.png')
    print("  已保存: test_hierarchy.png")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)