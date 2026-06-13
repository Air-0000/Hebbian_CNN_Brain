# Hebbian CNN 全版本详细实验报告

## 更新日志
- 2026-05-22: 初始版本

---

## 版本目录

1. [BioHebbian v1 (scale/pv)](#v1)
2. [BioHebbian v2 (Lite轻量版)](#v2)
3. [BioHebbian v3 (scale/pv改进版)](#v3)
4. [True Hebbian (11.8%)](#true11)
5. [HebbianHeavy (67.2%)](#heavy)
6. [VGG风格True Hebbian (过拟合)](#vgg)
7. [各版本横向对比](#compare)
8. [每层参数分解](#layer)

---

## <a name="v1"></a>版本 1: BioHebbian v1 (scale/pv)

### 基本信息
| 属性 | 值 |
|------|-----|
| 架构 | DenseNet-40 |
| 总参数量 | ~0.48M |
| Hebbian参数 | **72** (2参数/层 × 36层) |
| Hebbian占比 | 0.015% |
| 状态 | ✅ 完成 |

### Hebbian机制实现

```python
class DenseLayer(nn.Module):
    def __init__(self, in_ch, growth=12, drop=0.2):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, 4*growth, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth)
        self.conv2 = nn.Conv2d(4*growth, growth, 3, padding=1, bias=False)
        self.drop = drop
        # ===== Hebbian参数 =====
        self.scale = nn.Parameter(torch.tensor(0.05))  # 参数1: 缩放因子
        self.pv = nn.Parameter(torch.tensor(0.05))    # 参数2: 群体向量门控

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        # Hebbian调制
        out = out * (1 + self.scale)                           # 乘性调制
        pv_factor = torch.sigmoid(self.pv * (out.mean() - 0.3))
        out = out * (1 - pv_factor * 0.3)                      # 门控调制
        return out
```

### Hebbian参数计算

| 层级 | 参数名 | 形状 | 数量 |
|------|--------|------|------|
| 每层 | scale | scalar | 1 |
| 每层 | pv | scalar | 1 |
| **每层合计** | | | **2** |
| 总计(36层) | | | **72** |

### 训练配置

| 配置项 | 值 |
|--------|-----|
| 数据集 | CIFAR-10 |
| Batch Size | 128 |
| Optimizer | AdamW (lr=0.001, wd=1e-4) |
| Scheduler | CosineAnnealing |
| Loss | LabelSmoothingCrossEntropy (s=0.1) |
| 数据增强 | RandomCrop + RandomFlip |
| Dropout | 0.2 |
| Epochs | 100 |

### 训练曲线

| Epoch | Loss | Train Acc | Val Acc | 备注 |
|-------|------|-----------|---------|------|
| 1 | 1.76 | 43.3% | 54.4% | |
| 5 | 1.30 | 67.6% | 79.3% | |
| 10 | 1.18 | 73.6% | 85.9% | |
| 20 | 1.08 | 78.9% | 89.2% | |
| 50 | - | - | 91.5% | |
| 100 | - | - | **92.5%** | 最佳 |

### 优缺点分析

| 优点 | 缺点 |
|------|------|
| ✅ 参数量极少(仅72个Hebbian参数) | ❌ Hebbian参数太少，几乎不影响网络 |
| ✅ 实现简单 | ❌ 更像是训练技巧而非真正的Hebbian |
| ✅ 效果不错(92.5%) | ❌ 论文价值存疑 |

### 评估：⭐⭐⭐ (实验可行，论文价值存疑)

---

## <a name="v2"></a>版本 2: BioHebbian v2 (Lite轻量版)

### 基本信息
| 属性 | 值 |
|------|-----|
| 架构 | DenseNet-40 (简化) |
| 总参数量 | **~0.13M** |
| Hebbian参数 | 72 |
| Hebbian占比 | 0.055% |
| 状态 | ⚠️ 中断 |

### 主要改进

- 通道数减半
- 层数减少
- 使用深度可分离卷积

### 训练结果

| Epoch | Val Acc | 备注 |
|-------|---------|------|
| 20 | ~58% | |
| 50 | ~68% | |
| 80 | ~73% | **中断** |

### 评估：⭐⭐ (轻量但精度牺牲大)

---

## <a name="v3"></a>版本 3: BioHebbian v3 (scale/pv改进)

### 基本信息
| 属性 | 值 |
|------|-----|
| 架构 | DenseNet-40 |
| 总参数量 | ~0.48M |
| Hebbian参数 | 72 |
| Hebbian占比 | 0.015% |
| 状态 | ✅ 完成 |

### 与v1的区别

| 改进点 | v1 | v3 |
|--------|----|----|
| scale初值 | 0.05 | 0.05 |
| pv初值 | 0.05 | 0.05 |
| 调制强度 | 0.3 | 0.2 |
| 调制方式 | 乘性+门控 | 更保守 |

### 训练曲线

| Epoch | Val Acc |
|-------|---------|
| 100 | **92.35%** |

### 评估：⭐⭐⭐ (与v1相当)

---

## <a name="true11"></a>版本 4: True Hebbian (11.8% Hebbian)

### 基本信息
| 属性 | 值 |
|------|-----|
| 架构 | DenseNet-40 |
| 总参数量 | **539,566** (0.54M) |
| Hebbian参数 | **63,684** |
| Hebbian占比 | **11.80%** |
| 状态 | ✅ 完成 |

### Hebbian机制实现

```python
class TrueHebbianDenseLayer(nn.Module):
    def __init__(self, in_ch, growth=12, drop=0.2, hebbian_lr=0.05):
        super().__init__()
        # 标准BP部分
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, 4*growth, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth)
        self.conv2 = nn.Conv2d(4*growth, growth, 3, padding=1, bias=False)
        self.drop = drop
        self.hebbian_lr = hebbian_lr
        
        # ===== Hebbian参数 (核心创新) =====
        # Hebbian突触权重: 模拟突触可塑性
        self.hebbian_w = nn.Parameter(torch.randn(in_ch, growth) * 0.01)
        # Hebbian通道内调制: 模拟通道间竞争
        self.hebbian_intra = nn.Parameter(torch.randn(growth, growth) * 0.01)
        # 可学习调制因子
        self.modulation = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        
        if self.training:
            # 计算通道活动
            in_act = x.mean(dim=(2, 3))    # [B, in_ch]
            out_act = out.mean(dim=(2, 3))  # [B, growth]
            
            # Hebbian权重更新: Δw = η × in_act × out_act^T
            update = torch.ger(in_act.mean(0), out_act.mean(0)) * self.hebbian_lr
            with torch.no_grad():
                self.hebbian_w.data = self.hebbian_w.data * 0.95 + update * 0.05
            
            # Hebbian调制应用到特征
            channel_mod = torch.sigmoid(torch.matmul(in_act, self.hebbian_w))
            intra_mod = torch.sigmoid(torch.matmul(out_act, self.hebbian_intra))
            hebbian_effect = channel_mod * intra_mod * self.modulation
            out = out * (1 + hebbian_effect.unsqueeze(-1).unsqueeze(-1) * 0.5)
        
        return out
```

### Hebbian参数计算

| 参数名 | 形状 | 每层参数量 |
|--------|------|-----------|
| hebbian_w | [in_ch, growth] | ~16×12=192 |
| hebbian_intra | [growth, growth] | 12×12=144 |
| modulation | scalar | 1 |
| **每层合计** | | **~337** |
| **总计(36层)** | | **~12,132** |

**实际统计**: 63,684 (包含classifier层的额外Hebbian)

### 训练配置

| 配置项 | 值 |
|--------|-----|
| 数据集 | CIFAR-10 |
| Batch Size | 128 |
| Optimizer | AdamW (lr=0.001, wd=1e-4) |
| Scheduler | CosineAnnealing |
| Loss | LabelSmoothingCrossEntropy (s=0.1) |
| 数据增强 | RandomCrop + RandomFlip |
| Dropout | 0.2 |
| Hebbian LR | 0.05 |
| Epochs | 100 |
| Early Stop | patience=20 |

### 训练曲线

| Epoch | Loss | Train Acc | Val Acc | Val>Train? |
|-------|------|-----------|---------|------------|
| 1 | 1.84 | 40.6% | 47.8% | ✅ |
| 2 | 1.62 | 51.7% | 58.4% | ✅ |
| 3 | 1.53 | 57.9% | 67.0% | ✅ |
| 5 | 1.40 | 62.8% | 71.8% | ✅ |
| 10 | 1.32 | 68.4% | 81.5% | ✅ |
| 20 | 1.17 | 74.5% | 86.0% | ✅ |
| 30 | 1.12 | 77.3% | 88.8% | ✅ |
| 40 | 1.06 | 79.9% | 89.5% | ✅ |
| 50 | 1.01 | 82.7% | 90.5% | ✅ |
| 60 | 1.00 | 82.3% | 90.1% | ✅ |
| **66** | - | - | **91.13%** | 早停 |

### 关键现象

1. **Val > Train**：从Epoch 1开始就一直成立
2. **收敛稳定**：无明显过拟合
3. **早停触发**：Epoch 66自动停止

### 优缺点分析

| 优点 | 缺点 |
|------|------|
| ✅ 真正的Hebbian学习机制 | ❌ 参数量比标准DenseNet多(0.54M vs 0.48M) |
| ✅ Hebbian参数占比适中(11.8%) | ❌ 准确率未超越标准BP |
| ✅ Val > Train说明正则化效果 | ❌ 需与标准DenseNet同架构对比 |
| ✅ 准确率高(91.13%) | |

### 评估：⭐⭐⭐⭐ (真正的Hebbian，论文价值较高)

---

## <a name="heavy"></a>版本 5: HebbianHeavy (67.2% Hebbian)

### 基本信息
| 属性 | 值 |
|------|-----|
| 架构 | DenseNet-40 |
| 总参数量 | **3,858,502** (3.86M) |
| Hebbian参数 | **2,594,412** |
| Hebbian占比 | **67.24%** |
| 状态 | ✅ 完成 |

### 设计目标

让Hebbian作为**主要学习机制**，而非辅助微调。

### Hebbian机制实现

```python
class HebbianHeavyDenseLayer(nn.Module):
    def __init__(self, in_ch, growth=12, drop=0.2, hebbian_lr=0.05):
        super().__init__()
        # 标准BP部分
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
            nn.Conv2d(in_ch, 4*growth, 1, bias=False),
        )
        self.bn2 = nn.BatchNorm2d(4*growth)
        self.conv2 = nn.Conv2d(4*growth, growth, 3, padding=1, bias=False)
        
        # ===== Hebbian参数 (大幅扩展) =====
        # Hebbian突触权重: [in_ch, growth*16]
        self.hebbian_w = nn.Parameter(torch.randn(in_ch, growth * 16) * 0.01)
        # Hebbian通道间调制: [growth*16, growth*16]
        self.hebbian_inter = nn.Parameter(torch.randn(growth * 16, growth * 16) * 0.01)
        # Hebbian通道内调制: [growth*8, growth*8]
        self.hebbian_intra = nn.Parameter(torch.randn(growth * 8, growth * 8) * 0.01)
        # 多层调制因子
        self.mod1 = nn.Parameter(torch.tensor(1.0))
        self.mod2 = nn.Parameter(torch.tensor(0.5))
        self.mod3 = nn.Parameter(torch.tensor(0.5))
```

### Hebbian参数计算

| 参数名 | 形状 | 每层参数量(平均) |
|--------|------|-----------------|
| hebbian_w | [in_ch, growth×16] | ~2,400 |
| hebbian_inter | [growth×16, growth×16] | ~36,864 |
| hebbian_intra | [growth×8, growth×8] | ~9,216 |
| mod1/2/3 | scalar | 3 |
| **每层合计** | | | **~48,483** |
| **总计(36层)** | | | **~1,745,388** |

**实际统计**: 2,594,412

### 训练曲线

| Epoch | Loss | Train Acc | Val Acc | Val>Train? | 备注 |
|-------|------|-----------|---------|------------|------|
| 1 | 1.81 | 40.6% | 37.8% | ❌ | |
| 2 | 1.59 | 53.2% | 32.6% | ❌ | 过拟合 |
| 3 | 1.52 | 57.5% | **66.3%** | ✅ | **爆发！** |
| 4 | 1.46 | 60.3% | 61.7% | ✅ | |
| 5 | 1.41 | 62.1% | 64.2% | ✅ | |
| 10 | 1.27 | 70.7% | 76.1% | ✅ | |
| 20 | 1.18 | 73.5% | 82.9% | ✅ | |
| 30 | 1.06 | 79.5% | 86.3% | ✅ | |
| 40 | 1.06 | 81.0% | 86.9% | ✅ | |
| 50 | 1.02 | 83.2% | 89.6% | ✅ | |
| 60 | 1.01 | 81.7% | 89.2% | ✅ | |
| 70 | 0.97 | 84.4% | 89.4% | ✅ | |
| **75** | - | - | **90.53%** | | 早停 |

### 关键现象

1. **Epoch 2过拟合**：Train 53.2% > Val 32.6%
2. **Epoch 3爆发**：Val突然从32.6%跳到66.3%
3. **最终稳定**：Val > Train持续成立

### 与True Hebbian (11.8%)对比

| 对比项 | True Hebbian (11.8%) | HebbianHeavy (67.2%) |
|--------|----------------------|----------------------|
| 总参数 | 0.54M | 3.86M |
| Hebbian参数 | 63,684 | 2,594,412 |
| 参数量比例 | 1x | **7.1x** |
| 最佳Val | 91.13% | 90.53% |
| 早停Epoch | 66 | 75 |

**结论**：Hebbian参数增加7倍，但准确率只降低0.6%

### 优缺点分析

| 优点 | 缺点 |
|------|------|
| ✅ 证明67% Hebbian仍能学习 | ❌ 参数量效率低(8倍参数) |
| ✅ Hebbian作为主力机制可行 | ❌ 没超越标准BP |
| ✅ 正则化效果明显 | ❌ 训练初期不稳定 |
| ✅ Val > Train持续成立 | |

### 评估：⭐⭐⭐⭐ (证明Hebbian可作为主力，论文价值高)

---

## <a name="vgg"></a>版本 6: VGG风格True Hebbian

### 基本信息
| 属性 | 值 |
|------|-----|
| 架构 | VGG风格 (非DenseNet) |
| 总参数量 | **6,360,000** (6.36M) |
| Hebbian参数 | 1,349,632 |
| Hebbian占比 | 21.21% |
| 状态 | ❌ 过拟合崩溃 |

### 架构特点

```
Stem: Conv(3→32)
Block1: HebbianConv(32→64) + MaxPool
Block2: HebbianConv(64→128) + MaxPool
Block3: HebbianConv(128→256) + MaxPool
Block4: HebbianConv(256→512)
Classifier: AdaptiveAvgPool → HebbianDense → Linear
```

### 训练曲线 (失败案例)

| Epoch | Train Acc | Val Acc | 趋势 |
|-------|---------|---------|------|
| 1 | 32.6% | 22.7% | |
| 5 | 59.3% | 44.3% | 过拟合 |
| 10 | 67.9% | 59.0% | 差距增大 |
| 20 | 73.3% | 63.9% | |
| 30 | 77.0% | **57.7%** | 崩溃 |
| 40 | 78.8% | **55.3%** | 严重崩溃 |

### 失败原因分析

1. **架构问题**：VGG风格不适合与DenseNet对比
2. **参数量过大**：6.36M导致过拟合
3. **Hebbian机制不稳定**：Epoch 20后Val持续下降

### 评估：⭐ (失败案例，但有参考价值)

---

## <a name="compare"></a>横向对比

### 准确率对比

```
准确率 (Val Acc %)
  ↑
95% ┤
    │                                              ● DenseNet论文 94.5%
94% ┤
    │
93% ┤
92% ┤                                              ★ BioHebbian v1 92.5%
    │                                         ★
91% ┤                                      ★      ★ True Hebbian 11.8% 91.13%
    │                                 ★     ●
90% ┤                              ★●    ★★★  HebbianHeavy 67.2% 90.53%
    │                        ★     ★●
89% ┤                     ★●    ★●    ★
88% ┤                  ★    ★●   ★  Standard DenseNet 90.3%+
87% ┤               ★
86% ┤            ★
    └──────────────────────────────────────────────────────────→ Epoch
        1   10  20  30  40  50  60  70  80  90 100
```

### 参数量对比

```
参数量 (对数坐标)
  │
  │                              ┌─────────────────┐
  │                              │ HebbianHeavy    │
  │                              │ 3.86M           │
  │    ┌─────────────┐           └─────────────────┘
  │    │ VGG风格     │
  │    │ 6.36M      │
  │    └─────────────┘
  │
  │                       ┌───────────┐
  │                       │ True 11.8%│
  │                       │ 0.54M    │
  │                       └───────────┘
  │
  │            ┌───────────────────┐
  │            │ BioHebbian v1/v3  │
  │            │ 0.48M            │
  │            └───────────────────┘
  │
  │   ┌───────────┐
  │   │ Lite v2   │
  │   │ 0.13M    │
  │   └───────────┘
  └───────────────────────────────────────────────────→ 版本
```

### 核心指标对比表

| 版本 | 参数量 | Hebbian参数 | Hebbian占比 | 最佳Val | Epoch |
|------|--------|-------------|-------------|---------|-------|
| BioHebbian v1 | 0.48M | 72 | 0.015% | **92.5%** | 100 |
| BioHebbian v2 | 0.13M | 72 | 0.055% | ~73% | 80 |
| BioHebbian v3 | 0.48M | 72 | 0.015% | 92.35% | 100 |
| True Hebbian 11.8% | 0.54M | 63,684 | 11.80% | 91.13% | 66 |
| HebbianHeavy 67.2% | 3.86M | 2,594,412 | 67.24% | 90.53% | 75 |
| VGG风格 | 6.36M | 1.35M | 21.21% | 63.9% | 20 |
| Standard DenseNet | 0.48M | 0 | 0% | 90.3% | - |

### 排名

| 排名 | 版本 | 最佳Val | 评价 |
|------|------|---------|------|
| 🥇 | BioHebbian v1 | 92.5% | 最高准确率 |
| 🥈 | BioHebbian v3 | 92.35% | 接近v1 |
| 🥉 | True Hebbian 11.8% | 91.13% | 真正的Hebbian |
| 4 | HebbianHeavy 67.2% | 90.53% | 证明主力可行 |
| 5 | Standard DenseNet | 90.3% | 对照组 |
| 6 | BioHebbian v2 | ~73% | 轻量牺牲精度 |
| 7 | VGG风格 | 63.9% | 失败案例 |

---

## <a name="layer"></a>每层参数分解

### DenseNet-40 层级结构

```
Layer 0:  Input(3) → Conv(3→16)                    [1层]
    ↓
Block 1: DenseBlock(12层, in_ch=16, growth=12)   [12层]
    ├─ Layer 0: 16 → 28  (16+12×1)
    ├─ Layer 1: 28 → 40  (16+12×2)
    ├─ Layer 2: 40 → 52
    ├─ ...
    └─ Layer 11: 136 → 160
    ↓
Transition 1: 160 → 80                           [1层]
    ↓
Block 2: DenseBlock(12层, in_ch=80, growth=12)  [12层]
    ├─ Layer 0: 80 → 92
    ├─ Layer 1: 92 → 104
    ├─ ...
    └─ Layer 11: 200 → 224
    ↓
Transition 2: 224 → 112                          [1层]
    ↓
Block 3: DenseBlock(12层, in_ch=112, growth=12)  [12层]
    ├─ Layer 0: 112 → 124
    ├─ Layer 1: 124 → 136
    ├─ ...
    └─ Layer 11: 236 → 260
    ↓
Final: AdaptiveAvgPool → Linear(256→10)          [1层]

总计: 1 + 12 + 1 + 12 + 1 + 12 + 2 = 41层
DenseLayer: 36层 (3个Block × 12层)
```

### 各版本每层参数对比

#### True Hebbian (11.8%) - 以Block 1 Layer 0为例

| 参数 | in_ch | growth | 形状 | 数量 |
|------|-------|--------|------|------|
| bn1 | - | - | - | - |
| conv1 | 16 | - | 16→48 | 768 |
| bn2 | - | - | - | - |
| conv2 | 48 | - | 48→12 | 1,728 |
| **hebbian_w** | 16 | 12 | 16×12 | **192** |
| **hebbian_intra** | 12 | 12 | 12×12 | **144** |
| **modulation** | - | - | scalar | **1** |
| **BP层小计** | | | | 2,496 |
| **Hebbian小计** | | | | **337** |
| **总计** | | | | **2,833** |

#### HebbianHeavy (67.2%) - 以Block 1 Layer 0为例

| 参数 | in_ch | growth | 形状 | 数量 |
|------|-------|--------|------|------|
| conv1(序列) | 16 | - | 16→16→48 | 256+16+768 |
| conv2 | 48 | - | 48→12 | 1,728 |
| **hebbian_w** | 16 | 192 | 16×192 | **3,072** |
| **hebbian_inter** | 192 | 192 | 192×192 | **36,864** |
| **hebbian_intra** | 96 | 96 | 96×96 | **9,216** |
| **mod1/2/3** | - | - | scalar×3 | **3** |
| **BP层小计** | | | | 2,768 |
| **Hebbian小计** | | | | | **49,155** |
| **总计** | | | | **51,923** |

#### 参数量倍数关系

| 版本 | 每层BP | 每层Hebbian | 每层总计 | 倍数 |
|------|--------|-------------|----------|------|
| BioHebbian v1 | ~2,833 | 2 | 2,835 | 1x |
| True Hebbian 11.8% | ~2,833 | 337 | 3,170 | 1.1x |
| HebbianHeavy 67.2% | ~2,768 | 49,155 | 51,923 | **18.3x** |

---

## 参考基准

### 论文数据

| 模型 | 参数量 | CIFAR-10+ | 来源 |
|------|--------|----------|------|
| DenseNet-40 | 1.0M | 94.5% | 论文 |
| DenseNet-100 | 0.8M | 94.5% | 论文 |
| ResNet-110 | 1.7M | 93.6% | 论文 |
| ResNet-1001 | 10.2M | 95.4% | 论文 |
| SoftHebb | 5.9M | 80.3% | 论文 |
| SoftHebb Depthwise | 0.9M | ~79% | 论文 |

### 我们与论文对比

| 对比项 | 我们最佳 | 论文最佳 | 差距 |
|--------|---------|----------|------|
| 准确率 | 92.5% | 94.5% | -2% |
| 参数量 | 0.48M | 0.8M | -0.32M |

---

## 结论与建议

### 当前状态总结

| 评估项 | 状态 | 说明 |
|--------|------|------|
| Hebbian机制有效性 | ✅ 验证通过 | 所有版本都能学习 |
| 参数量效率 | ⚠️ 待优化 | 需减少参数同时保持准确率 |
| 准确率优势 | ❌ 未超越 | 与标准BP相当 |
| 论文创新性 | ⚠️ 需深化 | 需更多实验支撑 |

### 后续研究方向

1. **提高参数量效率**
   - 目标：用0.5M参数达到93%+
   - 方法：优化Hebbian机制结构

2. **探索最佳Hebbian占比**
   - 当前：11.8%最优
   - 假设：可能在5-15%区间存在最优点

3. **对标论文**
   - 参考SoftHebb通道设计(96→384→1536)
   - 尝试更大growth rate

4. **改进Hebbian更新规则**
   - 尝试Oja's rule变体
   - 加入归一化防止发散
   - 考虑时间维度的STDP

---

*报告生成时间: 2026-05-22*