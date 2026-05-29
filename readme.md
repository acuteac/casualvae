# ACLF 多模态因果 VAE 项目说明

本项目用于构建 ACLF/肝衰竭相关的多模态因果 VAE 模型，整合临床锚定变量、临床结局、蛋白组学和代谢组学数据，学习潜变量层因果结构，并将因果图投影到特征层，用于后续分子到临床结局的因果边分析和虚拟干预。

## 当前项目状态

当前主流程使用全特征数据：

- 样本总数：1950
- anchor 变量：11
- 连续临床结局：14
- 二分类临床结局：3
- 蛋白特征：2922
- 代谢物特征：252

数据已完成以下处理：

- 过滤 `Unnamed*` / `index` 等 CSV 过程列
- 过滤全缺失或常量 omics 特征
- 删除全缺失蛋白 `GLIPR1`
- 排除非数值 anchor `assess_date`
- 对 anchor、连续 phenotype、protein、metabolite 做 z-score 标准化
- 二分类结局转换为 `0/1` 浮点值
- 保留 NaN 作为缺失掩码来源，模型内部按 mask 计算重构损失

## 目录结构

```text
casualvae/
  data_augmentation.py
  prepare_augmented_structured.py
  prepare_full_augmented_structured.py
  generate_full_config.py
  module_aclf_partial_multimodal.py
  aclf_full_multimodal_config.json
  requirements.txt
  outputs/
    augmented/
    structured_full/
```

## 主要文件作用

### `data_augmentation.py`

生成或复制增强数据到 `outputs/augmented/`。这是当前结构化数据的上游数据源。

### `prepare_full_augmented_structured.py`

当前主数据准备脚本。它从 `outputs/augmented/` 读取数据，生成模型训练需要的结构化输入到 `outputs/structured_full/`。

输出包括：

- `anchors_structured.csv`：标准化后的临床锚定变量
- `phenotype_structured.csv`：标准化后的连续结局和二分类结局
- `protein_full_structured.csv`：标准化后的蛋白组学输入
- `metabolite_full_structured.csv`：标准化后的代谢组学输入
- `full_feature_columns.json`：蛋白和代谢物特征列名
- `data_quality_summary.json`：数据质量报告
- `preprocessing_params.json`：标准化均值和标准差

### `generate_full_config.py`

根据 `outputs/structured_full/full_feature_columns.json` 和结构化 CSV 表头生成训练配置 `aclf_full_multimodal_config.json`。

它会确保：

- 配置列名与实际 CSV 一致
- 不把 `Unnamed*` / `index` 过程列写入配置
- 不把非数值 anchor 写入配置
- 不把实际 CSV 中不存在的 phenotype 写入配置

### `aclf_full_multimodal_config.json`

当前全量模型训练配置，包含：

- anchor 列名
- phenotype 列名
- protein/metabolite 特征列名
- 可用性阈值
- 共享/私有潜变量维度
- 编码器隐藏层结构
- outcome head 结构

### `module_aclf_partial_multimodal.py`

核心模型和训练脚本，包含：

- `PartialMultiOmicsDataset`：多模态缺失数据 Dataset
- `SharedPrivateEncoder`：蛋白/代谢物共享-私有编码器
- `OutcomeEncoder` / `OutcomeDecoder`：结局变量编码器和解码器
- `BlockSCM`：潜变量层结构因果模型
- `PartialAnchoredCausalVAE`：完整模型
- 预训练、联合训练、loss、early stopping
- 潜变量因果图和特征层因果图导出

### `prepare_augmented_structured.py`

旧版结构化脚本，主要面向旧的小特征流程。当前全特征主流程优先使用 `prepare_full_augmented_structured.py`。

### `requirements.txt`

项目 Python 依赖列表。

## 模型结构概览

当前模型包含三个共享潜变量块：

```text
protein latent      -> 15 dimensions
metabolite latent   -> 15 dimensions
outcome latent      -> 3 dimensions
```

蛋白和代谢物还各有私有潜变量：

```text
protein private     -> 3 dimensions
metabolite private  -> 3 dimensions
```

因果结构在共享潜变量层学习。`outcome` 被建模为下游节点：

```text
protein / metabolite -> outcome
outcome -> protein / metabolite 被强制禁止
```

训练完成后，潜变量层因果图 `A` 会通过解码器投影到特征层：

```text
G = D @ A @ D.T
```

其中 `G_protein_to_outcome.csv` 和 `G_metabolite_to_outcome.csv` 是解释分子特征到临床结局影响的主要输出。

## 训练流程

训练分三阶段：

1. 蛋白自编码器预训练
2. 代谢物自编码器预训练
3. protein + metabolite + outcome 联合 SCM 训练

验证集比例默认是 `0.2`。训练集用于更新参数，验证集用于监控泛化表现、保存最佳模型和 early stopping。

## 常用命令

### 生成结构化全特征数据

```bash
python prepare_full_augmented_structured.py
```

### 生成全量训练配置

```bash
python generate_full_config.py
```

### 运行全量训练

```bash
python module_aclf_partial_multimodal.py ^
  --anchors_csv outputs/structured_full/anchors_structured.csv ^
  --phenotype_csv outputs/structured_full/phenotype_structured.csv ^
  --protein_csv outputs/structured_full/protein_full_structured.csv ^
  --metabolite_csv outputs/structured_full/metabolite_full_structured.csv ^
  --config_json aclf_full_multimodal_config.json ^
  --out_dir outputs/run_fixed_full ^
  --seed 42 ^
  --device cpu
```

如果机器有可用 CUDA，可以把 `--device cpu` 改成 `--device cuda`。

## 训练输出

训练输出目录通常包含：

- `model.pt`：模型权重
- `training_history.csv`：每个 epoch 的训练和验证指标
- `run_summary.json`：样本数、特征数、潜变量维度摘要
- `data_split_summary.json`：训练/验证样本划分摘要
- `latent_causal_A.csv`：潜变量层因果邻接矩阵
- `decoder_projection_D.csv`：解码器投影矩阵
- `feature_projection_G.csv`：完整特征层因果图
- `G_protein_to_outcome.csv`：蛋白到结局的因果边
- `G_metabolite_to_outcome.csv`：代谢物到结局的因果边
- `top_edges_protein_to_outcome.csv`：蛋白到结局 top 边
- `top_edges_metabolite_to_outcome.csv`：代谢物到结局 top 边
- `phenotype_cont_pred.csv`：连续结局预测
- `phenotype_bin_pred_prob.csv`：二分类结局预测概率
- `shared_latent_z.csv`：样本级共享潜变量
- `sample_availability.csv`：样本模态可用性

## 当前数据注意事项

- 联合训练真正依赖 protein 和 metabolite 都可用的 paired 样本，目前 paired 可用样本数记录在 `data_quality_summary.json`。
- 当前数据已经标准化，因此训练 loss 数值会比未标准化版本更可比较。
- 如果需要把预测值或效应解释回原始临床量纲，应使用 `preprocessing_params.json` 中的均值和标准差。
- 当前结果目录如 `outputs/run_fixed_full/` 是训练产物，不是源数据；重跑训练时应使用新的输出目录，避免覆盖重要结果。

