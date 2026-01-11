# 机械臂 C-Space NDF 性能优化技术报告

## 1. 优化目标
本报告旨在总结提高神经距离场 (Neural Distance Field, NDF) 在机械臂配置空间 (Configuration Space, C-Space) 中表示精度的技术方案。该方案在不增加原始采样数据量的前提下，通过增强模型的特征表达能力与物理一致性约束，显著提升了边界识别准确率，并为 6-DOF 乃至更高维度的扩展奠定了基础。

## 2. 核心方法

### 2.1 多尺度周期性嵌入 (Multi-scale Periodic Embedding)
机械臂的输入通常为关节角度 $\theta \in [-\pi, \pi]$。为了处理角度的周期性并捕捉 C-Space 边界的高频细节（如自碰撞的长条状区域），我们将原始输入通过多尺度嵌入层：

$$
PE(\theta) = \left[ \cos(k_1\theta), \sin(k_1\theta), \cos(k_2\theta), \sin(k_2\theta), \dots, \cos(k_L\theta), \sin(k_L\theta) \right]
$$

其中 $k_i \in \{1, 2, 4, 8, \dots \}$ 是选定的频率尺度。
*   **低频项**：保证全局形状的平滑连续。
*   **高频项**：允许网络学习更锐利的几何边缘，将边界准确率提升了约 5%。

### 2.2 Eikonal 梯度正则化 (Eikonal Regularization)
理想的 NDF 输出在边界附近的梯度模长应保持为 1，这不仅符合符号距离函数的物理定义，还能有效抑制边界抖动。我们在训练中引入了 Eikonal 正则化：

$$
\mathcal{L}_{eik} = \mathbb{E}_{\mathbf{q} \sim \mathcal{Q}} \left[ \left( \|\nabla_{\mathbf{q}} f(\mathbf{q})\|_2 - 1 \right)^2 \right]
$$

其中 $f(\mathbf{q})$ 是网络对配置 $\mathbf{q}$ 的预测输出。

### 2.3 模型容量提升 (Model Capacity)
针对自碰撞流形的复杂性，我们将神经网络的隐藏层宽度 (Hidden Dimension) 从 128 提升至 256（针对 3-DOF）或 512（建议 6-DOF）。这提供了足够的参数空间来拟合非线性的物理碰撞边界。

---

## 3. 损失函数 (Loss Functions)

训练过程采用组合损失函数，涵盖了分类符号约束、边缘间隔约束以及物理梯度约束。

### 3.1 混合 Kantorovich-Rubinstein (HKR) 损失
用于区分碰撞空间 $C_{obs}$ 和自由空间 $C_{free}$。规定碰撞点输出为负，自由空间点输出为正。

$$
\mathcal{L}_{HKR} = \underbrace{\mathbb{E}_{\mathbf{q} \in C_{obs}}[f(\mathbf{q})] - \mathbb{E}_{\mathbf{q} \in C_{free}}[f(\mathbf{q})]}_{\text{KR Term (符号对齐)}} + \lambda \cdot \underbrace{\mathcal{L}_{hinge}}_{\text{边界间隔控制}}
$$

### 3.2 Hinge 损失项
为了确保模型在边界处有清晰的决定平面，引入 Margin $m$：

$$
\mathcal{L}_{hinge} = \mathbb{E}_{\mathbf{q} \in C_{obs}}[\max(0, m + f(\mathbf{q}))] + \mathbb{E}_{\mathbf{q} \in C_{free}}[\max(0, m - f(\mathbf{q}))]
$$

### 3.3 总训练损失
$$
\mathcal{L}_{total} = \mathcal{L}_{HKR} + \alpha \cdot \mathcal{L}_{eik}
$$

其中 $\alpha$ 是梯度正则化的权重（推荐取值 $10^{-4}$ 至 $10^{-3}$）。

---

## 4. 实验结论与 6-DOF 建议

通过上述优化，我们在 3-DOF Jaco 模型上获得了 **98.7%** 的全局准确率和 **87.3%** 的边界准确率。

**针对 6-DOF 扩展的建议配置：**
1.  **嵌入频率**：使用 `[1, 2, 4, 8]` 四个尺度。
2.  **网络宽度**：使用 `512` 隐藏层神经元。
3.  **梯度约束**：始终保持开启 $\mathcal{L}_{eik}$，以防止高维空间中出现预测孤岛或空洞。
