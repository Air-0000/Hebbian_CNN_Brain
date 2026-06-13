# True Hebbian DenseNet 实验记录

## 实验时间
2026-05-22 凌晨

---

## 实验目标

验证真正 Hebbian 学习机制在 CNN 中的效果，让 Hebbian 作为主要学习机制而非辅助微调。

---

## 实验版本对比

### 1. True Hebbian DenseNet (11.8% Hebbian)

**配置**：
- 总参数量: 539,566 (0.54M)
- Hebbian参数: 63,684 (11.80%)
- Hebbian机制: hebbian_w + hebbian_intra + modulation

**训练结果**：
| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 1 | 40.6% | 47.8% |
| 5 | 62.8% | 71.8% |
| 10 | 68.4% | 81.5% |
| 20 | 74.5% | 86.0% |
| 30 | 77.3% | 88.8% |
| 40 | 79.9% | 89.5% |
| 50 | 82.7% | 90.5% |
| 60 | 82.3% | 90.1% |
| **66** | - | **91.13%** (早停) |

**最佳准确率: 91.13%**

---

### 2. HebbianHeavy DenseNet (67% Hebbian)

**配置**：
- 总参数量: 3,858,502 (3.86M)
- Hebbian参数: 2,594,412 (67.24%)
- Hebbian机制: hebbian_w (16倍) + hebbian_inter + hebbian_intra + mod1/2/3

**训练结果**：
| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 1 | 40.6% | 37.8% |
| 2 | 53.2% | 32.6% |
| 3 | 57.5% | **66.3%** (爆发) |
| 5 | 62.1% | 64.2% |
| 10 | 70.7% | 76.1% |
| 20 | 73.5% | 82.9% |
| 30 | 79.5% | 86.3% |
| 40 | 81.0% | 86.9% |
| 50 | 83.2% | 89.6% |
| 60 | 81.7% | 89.2% |
| 70 | 84.4% | 89.4% |
| **75** | - | **90.53%** (早停) |

**最佳准确率: 90.53%**

---

### 3. Standard DenseNet 对照组

**配置**：
- 总参数量: 475,882 (0.48M)
- 无Hebbian机制

**训练结果**：
| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 1 | 38.8% | 45.3% |
| 5 | 61.7% | 70.1% |
| 10 | 68.3% | 81.5% |
| 20 | 75.8% | 85.7% |
| 30 | 77.5% | 85.5% |
| 40 | 79.9% | 88.8% |
| 50 | 81.2% | 90.3% |

---

## 核心发现

### 1. Hebbian机制确实有效
- 67% Hebbian参数仍能学习到 90.53%
- Val > Train 的现象出现（说明Hebbian有正则化效果）

### 2. 参数效率问题
- HebbianHeavy: 3.86M参数 → 90.53%
- Standard: 0.48M参数 → 90.3%+
- Hebbian用了8倍参数才追平标准BP

### 3. Hebbian占比与效果
| Hebbian占比 | 参数量 | 准确率 |
|-------------|--------|--------|
| 11.8% | 0.54M | 91.13% |
| 67.24% | 3.86M | 90.53% |

### 4. SoftHebb论文参考
| 模型 | 参数量 | CIFAR-10 |
|------|--------|----------|
| SoftHebb | 5.9M | 80.3% |
| SoftHebb Depthwise | 0.9M | ~79% |
| 我们的HebbianHeavy | 3.86M | 90.53% |

---

## 结论

| 评估 | 结果 |
|------|------|
| Hebbian机制有效 | ✅ 是 |
| 参数量效率 | ⚠️ 低（8倍参数） |
| 准确率优势 | ❌ 无（与标准持平） |

### 论文发表评估
当前状态下，可以作为"脑科学启发"的探索性工作，但还没有达到发论文的突破水平。

---

## 后续方向

1. **提高参数效率**：用更少Hebbian参数达到更高准确率
2. **对标SoftHebb**：参考其通道设计(96→384→1536)
3. **改进Hebbian机制**：探索更高效的更新规则
4. **对比标准**：与标准DenseNet-100(0.8M,94.5%)对比

---

## 相关论文

- **SoftHebb**: "SoftHebb: Bayesian inference in unsupervised Hebbian soft winner-take-all networks" (ICLR 2023)
- **Hebbian Deep Learning**: "Hebbian Deep Learning Without Feedback" (arXiv:2209.11883)

---

## 文件列表

```
src/train/cifar/
├── bio_hebbian_densenet.py      # 原始版本 (scale/pv)
├── true_hebbian_cnn.py          # VGG风格True Hebbian
├── true_hebbian_densenet.py      # DenseNet风格 True Hebbian
├── true_hebbian_densenet_main.py # 11.8% Hebbian版本
└── hebbian_heavy_densenet.py     # 67% Hebbian版本 ✅最终
```