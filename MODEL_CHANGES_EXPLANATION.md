# 模型改动说明文档

## 概述

本次改动的核心目标是将 ACLF 部分多模态锚定因果 VAE 模型从"使用筛选后的数据"扩展为"使用原始全特征数据"，并将**结局变量纳入因果图作为最下游节点**，使得因果图能够直接展示蛋白质/代谢物对临床结局的因果效应。

---

## 一、改动前后对比

### 1.1 数据维度对比

| 数据类型 | 改动前 | 改动后 |
|---------|--------|--------|
| 蛋白质 | 32 列（领域专家筛选） | ~2000 列（全部原始蛋白质） |
| 代谢物 | 40 列（相关性筛选） | ~250 列（全部原始代谢物） |
| 锚点变量 U | 12 列 | 12 列（不变） |
| 结局变量 Y | 15 连续 + 3 二分类 | 15 连续 + 3 二分类（不变） |

### 1.2 因果图结构对比

**改动前：**
```
蛋白质潜变量 ←→ 代谢物潜变量
        ↓
    [独立 MLP 头]
        ↓
     结局预测（不在因果图中）
```

**改动后：**
```
蛋白质潜变量 ←→ 代谢物潜变量
       ↓             ↓
       └──→ 结局潜变量 ←──┘
       （结局为纯下游节点）
```

---

## 二、新增组件详解

### 2.1 `OutcomeEncoder` 类（结局编码器）

**位置：** `module_aclf_partial_multimodal.py` 第 248 行附近

**作用：** 将临床结局变量（连续型 + 二分类）编码为共享潜变量。

**结构：**
- 输入：`y_cont`（连续结局）+ `y_bin`（二分类结局）拼接
- 主干：MLP（默认隐藏层 [64, 32]）
- 输出：`shared_mu`、`shared_logvar`（用于重参数化）

**设计选择：仅共享潜变量，无私有潜变量**
- 理由：结局是观测目标而非输入特征，不需要分离"私有干扰信息"
- 私有潜变量主要用于建模与因果机制无关的噪声，而结局本身即是要建模的目标

### 2.2 `OutcomeDecoder` 类（结局解码器）

**位置：** `module_aclf_partial_multimodal.py` 第 271 行附近

**作用：** 从结局共享潜变量 `z_outcome` 重构连续型和二分类结局。

**结构：**
- 输入：`z_outcome`（结局共享潜变量）
- 简单线性投影：`nn.Linear(shared_dim, n_cont + n_bin)`
- 输出：连续型预测 + 二分类 logits

**为什么使用线性而非 MLP？**
- 与蛋白质/代谢物的 `LinearSharedPrivateDecoder` 保持一致
- 便于通过解码器权重计算特征级因果图（`g = d @ a @ d.T`）
- 线性映射使因果图的边权重具有清晰的可解释性

### 2.3 `BlockSCM` 类的扩展

**主要修改：**

#### (1) 邻接矩阵掩码增强
```python
if self.has_outcome:
    o = self.block_slices["outcome"]
    p = self.block_slices["protein"]
    m = self.block_slices["metabolite"]
    # 强制 outcome 不能指向 protein 或 metabolite
    mask[p, o] = 0.0
    mask[m, o] = 0.0
```
**效果：** 因果流向只能是 `protein/metabolite → outcome`，禁止反向因果。

#### (2) 初始化 logits 调整
为 `protein → outcome` 和 `metabolite → outcome` 边设置初始值，鼓励模型探索这些边。

#### (3) 稀疏性惩罚扩展
新增三个块的稀疏性权重：
- `protein_to_outcome`: 0.5（中等稀疏）
- `metabolite_to_outcome`: 0.5（中等稀疏）
- `outcome_to_outcome`: 999.0（极强稀疏，强制结局节点之间无边）

---

## 三、训练流程详解

### 3.1 数据流

```
[原始增强数据]
     ↓
[prepare_full_augmented_structured.py]
     ↓
4 个结构化 CSV 文件
     ↓
[PartialMultiOmicsDataset]
     ↓
  3 个 DataLoader：
    - protein_train/val（蛋白质可用样本）
    - metabolite_train/val（代谢物可用样本）
    - paired_train/val（蛋白质和代谢物都可用）
```

### 3.2 三阶段训练

#### **阶段 1：蛋白质自编码器预训练**
- **数据：** 仅蛋白质可用样本
- **训练：** `enc_p` + `dec_p`
- **损失：** 蛋白质重构损失 + KL + 独立性 + 正交性
- **目的：** 学习蛋白质的有效低维表示

#### **阶段 2：代谢物自编码器预训练**
- **数据：** 仅代谢物可用样本
- **训练：** `enc_m` + `dec_m`
- **损失：** 代谢物重构损失 + KL + 独立性 + 正交性
- **目的：** 学习代谢物的有效低维表示

#### **阶段 3：联合 SCM 训练（关键阶段）**
- **数据：** 配对样本（蛋白质和代谢物都可用）
- **训练：** 全部参数（包括新增的 `enc_outcome` 和 `dec_outcome`）
- **流程：**
  1. 编码三种模态：
     ```
     enc_p = encoder_p(x_p, mask_p)
     enc_m = encoder_m(x_m, mask_m)
     enc_outcome = encoder_outcome(y_cont, y_bin)
     ```
  2. 拼接共享潜变量噪声：
     ```
     eps_shared = [enc_p.eps, enc_m.eps, enc_outcome.eps]
     ```
  3. SCM 求解（带掩码的邻接矩阵 A）：
     ```
     z_all = (I - A^T)^(-1) (eps_shared + anchor_context)
     ```
  4. 分离潜变量：z_p, z_m, z_outcome
  5. 解码重构：
     ```
     xhat_p = dec_p(z_p, private_p)
     xhat_m = dec_m(z_m, private_m)
     y_cont_hat, y_bin_logits = dec_outcome(z_outcome)
     ```

### 3.3 联合训练损失函数

```
total_loss = recon_p + recon_m                    # 蛋白质 & 代谢物重构
           + λ_outcome_cont × MSE(y_cont)         # 连续结局重构
           + λ_outcome_bin × BCE(y_bin)           # 二分类结局重构
           + λ_kl × KL(p) + KL(m) + KL(outcome)   # KL 散度（含结局）
           + λ_kl_priv × KL_private               # 私有潜变量 KL
           + λ_ind × 独立性惩罚                    # 共享 vs 私有解耦
           + λ_dag × DAG 惩罚                      # 仅在 protein/metabolite 块内
           + λ_sparse × 块稀疏性惩罚               # 含 outcome 块
           + λ_stable × 谱半径惩罚                 # SCM 稳定性
           + λ_ortho × 正交性惩罚（含 dec_outcome）
           + λ_dec_sparse × 解码器稀疏性（含 dec_outcome）
```

### 3.4 损失权重调度（Ramp Schedule）

为了训练稳定性，多个损失项采用渐进式激活：

| 损失项 | 起始 epoch | Ramp 时长 |
|--------|-----------|----------|
| KL 散度 | 0 | 60 epoch |
| 独立性 | warmup/2 = 10 | 60 epoch |
| DAG 惩罚 | warmup = 20 | 60 epoch |
| 稀疏性 | warmup = 20 | 60 epoch |
| 稳定性 | warmup = 20 | 60 epoch |

**为什么有 warmup？** 让模型先学习重构（基础任务），再逐步加入因果结构约束。

---

## 四、关键设计决策

### 4.1 为什么结局必须是纯下游节点？

**因果合理性：**
- 在临床场景中，蛋白质和代谢物的变化是疾病机制的体现
- 结局（如肝衰竭指标）是这些机制的"表现"，而非"原因"
- 强制 outcome → protein/metabolite 边为零，符合医学常识

**实现方式：**
1. **硬掩码**：在 `BlockSCM.__init__` 中将 `mask[p, o] = mask[m, o] = 0.0`
2. **软稀疏**：`outcome_to_outcome` 权重设为 999.0

### 4.2 为什么使用掩码而非数据筛选？

**优势：**
- 保留全部信息，模型自动学习哪些特征重要
- 通过特征级掩码处理缺失值，不强制丢弃样本
- 增加可发现的潜在因果关系数量

**代价：**
- 输入维度大幅增加（需要更大的网络）
- 训练时间和内存需求增加

**应对：**
- 蛋白质编码器从 [128, 64] 扩大到 [512, 256, 128]
- 代谢物编码器从 [128, 64] 扩大到 [256, 128, 64]
- 加强正则化（解码器稀疏性、正交性）

### 4.3 为什么因果图 G 用 `D @ A @ D^T`？

**数学解释：**
- `A`：潜变量层因果邻接矩阵（低维，可解释）
- `D`：解码器投影矩阵（潜变量 → 特征）
- `G = D @ A @ D^T`：将潜变量因果关系投影到特征空间

**意义：**
- `G[i, j]`：特征 j 对特征 i 的因果效应大小
- 包含结局后，`G[outcome_feature, protein_feature]` 直接给出**蛋白质对临床指标的因果效应**

---

## 五、输出文件说明

训练完成后，`out_dir` 包含以下关键文件：

### 5.1 模型文件
- `model.pt`：完整模型权重
- `used_config.json`：训练使用的配置
- `training_history.csv`：每个 epoch 的所有指标

### 5.2 潜变量层因果图
- `latent_causal_A.csv`：邻接矩阵 A，包含 outcome 节点
  - 行/列名：P1-P6（蛋白质潜变量）、M1-M6（代谢物潜变量）、O1-O6（结局潜变量）
- `decoder_projection_D.csv`：解码器投影矩阵
- `shared_latent_z.csv`：每个样本的潜变量值

### 5.3 特征层因果图
- `feature_projection_G.csv`：完整特征级因果图（包含结局特征）
- `G_protein_to_protein.csv`：蛋白质内部因果边
- `G_metabolite_to_metabolite.csv`：代谢物内部因果边
- `G_protein_to_metabolite.csv`：蛋白质 → 代谢物的边
- `G_metabolite_to_protein.csv`：代谢物 → 蛋白质的边
- **`G_protein_to_outcome.csv`**：蛋白质 → 结局的边 ⭐ 新增
- **`G_metabolite_to_outcome.csv`**：代谢物 → 结局的边 ⭐ 新增

### 5.4 Top 边排序
- `top_edges_*.csv`：每个块的 Top 50 最强因果边
- **`top_edges_protein_to_outcome.csv`** ⭐ 新增
- **`top_edges_metabolite_to_outcome.csv`** ⭐ 新增

### 5.5 预测结果
- `phenotype_cont_pred.csv`：连续结局预测值
- `phenotype_bin_pred_prob.csv`：二分类结局预测概率
- `sample_availability.csv`：每个样本的数据可用性

---

## 六、使用流程

### 步骤 1：准备数据
```bash
cd d:/working space/casualmodule/casualvae
python prepare_full_augmented_structured.py
```
**输出：** `outputs/structured_full/` 目录，包含 4 个 CSV 文件 + `full_feature_columns.json`

### 步骤 2：生成配置文件
```bash
python generate_full_config.py
```
**输出：** `aclf_full_multimodal_config.json`

### 步骤 3：训练模型
```bash
python module_aclf_partial_multimodal.py \
    --anchors_csv outputs/structured_full/anchors_structured.csv \
    --phenotype_csv outputs/structured_full/phenotype_structured.csv \
    --protein_csv outputs/structured_full/protein_full_structured.csv \
    --metabolite_csv outputs/structured_full/metabolite_full_structured.csv \
    --config_json aclf_full_multimodal_config.json \
    --out_dir results_full_with_outcome \
    --seed 42
```

### 步骤 4：分析结果
重点关注：
- `G_protein_to_outcome.csv` 和 `G_metabolite_to_outcome.csv`：找出对结局影响最大的蛋白质/代谢物
- `top_edges_*_to_outcome.csv`：Top 边排序，便于撰写论文/报告
- `latent_causal_A.csv`：检查潜变量层的因果结构是否合理

---

## 七、常见问题与调试

### Q1：训练损失不下降
**可能原因：**
- 学习率过大（高维输入下）
- 编码器容量不足

**解决：**
- `--lr_joint 5e-4`（默认 7e-4）
- 检查 `hidden_dims` 是否足够大

### Q2：因果图过于稠密
**可能原因：** 稀疏性惩罚不足

**解决：**
- 增大 `lambda_sparse`
- 增大 `protein_to_outcome` 和 `metabolite_to_outcome` 的块权重

### Q3：结局预测准确率低
**可能原因：** 通过 SCM 路径预测时损失了直接信息

**解决：**
- 增大 `lambda_outcome_cont` 和 `lambda_outcome_bin`
- 检查 `outcome_to_outcome` 是否过强（导致结局间真实关联无法建模）

### Q4：内存不足
**可能原因：** 蛋白质特征数过多（~2000）

**解决：**
- 减小 `batch_size`（默认 32 → 16）
- 在数据准备阶段过滤极低可用性特征（< 5%）

---

## 八、关键文件清单

### 新增文件
1. **`prepare_full_augmented_structured.py`**：数据准备脚本
2. **`generate_full_config.py`**：配置文件生成脚本
3. **`aclf_full_multimodal_config.json`**：完整特征配置（运行 generate 后生成）
4. **`MODEL_CHANGES_EXPLANATION.md`**：本说明文档

### 修改文件
1. **`module_aclf_partial_multimodal.py`**：核心模型代码
   - 新增 `OutcomeEncoder` 类（约第 248 行）
   - 新增 `OutcomeDecoder` 类（约第 271 行）
   - 修改 `BlockSCM` 类（支持 outcome 节点）
   - 修改 `PartialAnchoredCausalVAE` 类（集成结局编码器/解码器）
   - 修改 `forward_joint` 方法（新增 y_cont, y_bin 参数）
   - 修改 `joint_loss` 函数（添加结局 KL 和正则项）
   - 修改 `extract_outputs` 函数（导出含结局的因果图）

### 保持不变的文件
- `prepare_augmented_structured.py`：原数据准备脚本（处理筛选后的数据）
- `aclf_partial_multimodal_bundle/`：原版本的训练结果

---

## 九、原理总结

本次改动的核心创新点：

1. **数据驱动的特征选择**：不再依赖领域专家筛选，让模型从全部特征中自动学习。
2. **统一的因果框架**：将分离的"SCM + 独立 MLP 头"合并为"扩展 SCM"，结局成为因果图的有机组成部分。
3. **可解释的因果路径**：通过 `G_*_to_outcome.csv` 直接读出生物分子对临床结局的因果效应大小。
4. **结构化先验约束**：通过邻接矩阵掩码强制结局为纯下游节点，符合医学因果直觉。

最终得到的因果图可以回答这样的问题：
> "哪些蛋白质和代谢物对肝衰竭指标具有最强的因果影响？"
