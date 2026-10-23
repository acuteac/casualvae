# 4.4.1 候选干预对象与实验方案准备

此脚本只准备标准化输入特征的扰动计划，不加载权重、不调用 Encoder/SCM/OutcomeDecoder、不训练、不计算反事实、ΔOutcome、ATE 或剂量响应。输入特征位移不能直接称为严格 SCM `do` 干预，也不构成新因果靶点证据。

## 运行

在仓库根目录运行（已有 `casualvae_01` 环境中的 numpy、pandas、scipy 即可）：

```powershell
conda run --no-capture-output -n casualvae_01 python scripts/analysis/prepare_virtual_perturbation.py --help
conda run --no-capture-output -n casualvae_01 python scripts/analysis/prepare_virtual_perturbation.py --dry-run
conda run --no-capture-output -n casualvae_01 python scripts/analysis/prepare_virtual_perturbation.py
```

完整可配置示例：

```powershell
conda run --no-capture-output -n casualvae_01 python scripts/analysis/prepare_virtual_perturbation.py `
  --direct-dir artifacts/analysis/direct_effect_screening `
  --total-dir artifacts/analysis/total_effect_propagation `
  --data-dir data/structured_full `
  --config configs/aclf_full_multimodal_config.json `
  --primary-dir artifacts/multiseed/fixed_split/model_seed_456 `
  --replication-dirs artifacts/multiseed/variable_split/seed_456 artifacts/multiseed/variable_split/seed_789 `
  --sensitivity-dirs artifacts/multiseed/fixed_split/model_seed_789 `
  --protein-top-k 10 --metabolite-top-k 10 `
  --controls-per-modality 3 --protein-controls 3 --metabolite-controls 3 `
  --doses -1.0 -0.5 0.0 0.5 1.0 `
  --support-quantiles 0.01 0.99 --seed 42 `
  --out-dir artifacts/analysis/virtual_perturbation/phase1
```

退出码 0 表示准备成功或 dry-run 验证成功，2 表示输入/选项错误。缺失文件会集中列出。缺失历史 split EID 或 seed 元数据时必须找回原训练记录，不能由人数、目录名、当前 CSV 行序推测。脚本不会写入输入目录、其父/子目录、模型目录或源代码目录。

正常运行成功才发布四个 CSV、`experiment_config.json` 和 `preflight_report.json`。dry-run 仅写预检报告，不发布正式清单。失败仅更新 `status=blocked` 报告，可能包含明确标注的候选诊断预览及通过审计模型的样本统计；不会发布部分正式实验。旧正式文件如已存在会保留，但新报告未就绪时均不得执行。报告先写 `checking`，完成后才写 `ready`；各文件经临时文件替换发布。

## 固定选择规则

复用 4.3 的绝对效应排名函数和跨模型支持函数，4.2/4.3 的跨 Outcome 汇总函数、4.3 对 Direct 结果的一致性核对函数。模型注册表沿用 4.3 的 `primary`、`variable456`、`variable789`、`fixed789` 标签；自定义目录必须与输入分析 summary 的标签/角色/路径对应。

所有 17 个 Outcome 均参与筛选。在每个 **modality × Outcome** 内，按 `abs(total_effect)` 降序、原始 feature 升序排名。验证输入已有排名，不重新定义结果。每个模态分别按下列六元组字典式排序，无加权综合分数：

1. Primary Total Top20 跨 Outcome 次数：降序。
2. Primary Total Top50 跨 Outcome 次数：降序。
3. Primary Total 排名在全部 Outcome 的均值：升序。
4. 该特征 Primary Top20 边的四模型 Top20 支持数均值：降序。如果没有 Primary Top20 边，使用该特征达到最佳 Primary 排名的边。
5. 全部 Outcome 中 Total 方向与 Primary 一致的其他模型比例的均值：降序。零效应使用 `sign(0)=0`。
6. 原始 feature 编码：字典序升序。

Primary 出现次数范围 0–17，同一条边四模型支持数范围 0–4，方向一致比例的分母为其他三个模型，三者独立保存。代谢物用原始编码连接，名称仅从现有导出表取得并交叉核对，允许不同编码有相同名称，不使用名称排序或注入生物学知识。

每个模态先排除正式候选。剩余特征按其跨 Outcome **最优排名百分位**降序，再按平均排名百分位降序、编码升序，取后效应较低的候选池，池大小为 `max(对照数, ceil(剩余数 × 0.2))`。在按编码排序的池中无放回采样；随机流为 `SeedSequence([seed, modality_index])`，模态顺序 Protein=0、Metabolite=1。对照编号按采样后编码升序赋值。若剩余数量不足则报错，不静默减量。这是相对低结构效应对照，不是真实生物学阴性。

## 患者与剂量

遵循现有加载规则：anchors INNER JOIN phenotype，再 LEFT JOIN protein/metabolite，均按唯一整数 `eid` 连接，最终按 EID 排序。重复 ID、重复列、缺失配置特征、错误模态、非有限效应/无穷输入、浮点溢出、结果排名不符均报错。输入保持现有标准化尺度，不再次标准化。模态缺失掩码按配置特征顺序转换 float32 后的有限性计算，与 `PartialMultiOmicsDataset` 一致；不填补候选分子缺失值。

每模型读取六组 `*_train_eids`/`*_val_eids`，验证 EID 存在、同阶段训练/验证不交叠、两集合并集与该模态可用集合一致，以及人数、seed、有效模型配置、`best_tau` 一致。不同训练阶段之间允许重叠，不能将 paired validation 自动等同于从未训练。

- 主要队列：paired validation、双模态可用、具体候选已观测。
- 严格未训练标记：全部已连接 EID 减去 protein/metabolite/paired 三组训练 EID 并集；同时报告有任一模态、paired 可用及主队列中的严格未训练人数。全无模态患者可能使总体数字很大，不能当作有效独立评估人数。
- 探索性队列：该候选所属单模态可用且在该模态 validation，另一模态不可用，且未参与 paired training。单独标记并禁用联合执行，未来需要独立执行器。

每位主/探索队列患者都保留剂量行，候选缺失的行也保留且禁用；所有其他患者在 cohort 中保留审计记录，不生成剂量。支持范围来自 **对应模态预训练 EID 与 paired training EID 并集**中该特征的观测值；不使用仅验证样本，不把另一模态单独训练 EID 的特征加入支持范围。分位数使用 linear 插值，默认 1%/99%。没有训练观测时所有剂量均禁用并警告；仅 1 个观测也警告。

`target_value=baseline_value+dose_sd`，不截断。非零越界剂量禁用。0 SD 始终保留；若患者合格、有观测和训练支持，即使基线本身越界也允许作为配对基线，且保留 `out_of_support=true`，不将其解释为外推预测。缺失候选/缺失支持时该标志为空，不能解释为在支持范围内。

## 输出字段（schema_version = 4.4.1.v1）

CSV 布尔为 `True`/`False`，空数值为缺失/不适用，原因无异常为字符串 `none`。feature/EID 应按编码读取，不用展示名作为键。`experiment_config.json.outputs` 逐文件记录稳定列顺序、行数、SHA256。

### candidate_manifest.csv

每行一个候选/对照，唯一键 `modality,feature`。

| 字段 | 定义 |
|---|---|
| modality, feature, feature_display | Protein/Metabolite、原始分子编码、展示名（蛋白同编码） |
| candidate_role, rank | `candidate` 或 `low_structural_effect_control`；模态×角色内 1 起编号，对照编号不表示强效应顺序 |
| primary_top20_outcome_count, primary_top50_outcome_count | Primary 的 Total 跨 Outcome 次数 |
| primary_mean_total_rank, primary_best_total_rank | 全 Outcome 平均/最佳 Total 排名 |
| high_rank_mean_top20_model_support | 上述高排名边的四模型 Top20 支持均值 |
| mean_total_sign_agreement | 全 Outcome 其他三模型 Total 与 Primary 同号比例均值 |
| primary_direct_top20_outcome_count, primary_mean_direct_rank | Direct Top20 跨 Outcome 次数与平均 Direct 排名 |
| mean_abs_direct_effect, mean_abs_total_effect, mean_abs_indirect_effect | 全 Outcome 三类结构效应的绝对值均值 |
| direction_reversal_outcome_count | Primary Direct 与 Total 异号的 Outcome 数 |
| best_total_rank_percentile, mean_total_rank_percentile | `rank/该模态特征数` 的跨 Outcome 最小值/均值；对照池依据 |
| selection_reason | 固定筛选规则或对照采样原因 |

### candidate_outcome_evidence.csv

每行 `modality,feature,outcome`，另有 `feature_display,candidate_role`。对四模型标签 `{label}` 均保存 `{label}_direct_effect`、`{label}_indirect_effect`、`{label}_total_effect`、`{label}_direct_outcome_rank`、`{label}_total_outcome_rank`、`{label}_rank_shift`（Direct rank − Total rank）。

其余列为 `total_top20_support_count`（四模型数）、`total_sign_agreement_count`（其他三模型数）、`total_sign_agreement_fraction`、`direct_top20_support_count`、`direct_sign_agreement_fraction`、`primary_direction_reversal`、`primary_effect_pattern`。带符号效应保留方向证据。

### cohort_manifest.csv

每行 `model,eid,modality,feature`；覆盖全部 EID × 选中分子 × 模型。

| 字段 | 定义 |
|---|---|
| model, model_role, eid, modality, feature, candidate_role | 模型注册标签、角色、原始整数 EID、分子与角色 |
| protein_observed_fraction, metabolite_observed_fraction | 配置特征中有限值比例 |
| protein_available, metabolite_available, paired_available | 两模态阈值资格与两者交集 |
| candidate_observed, candidate_missing, baseline_value | 具体分子有限性/缺失、float32 模型输入的标准化基线值 |
| protein_train, protein_val, metabolite_train, metabolite_val, paired_train, paired_val | 六组原始划分归属 |
| not_joint_trained, strictly_untrained | 未参与 paired training；未参与任何训练阶段（不同概念） |
| cohort_type | `main_paired_validation`、`exploratory_single_modality_validation` 或 `excluded`；未根据候选缺失改写队列来源 |
| eligible_main, eligible_exploratory, eligible | 主/探索队列具体分子合格、两者并集；eligible 不等同于联合执行许可 |
| exclusion_reason | 非验证来源、模态不足、分子缺失等原因，多个原因以分号连接 |

### perturbation_plan.csv

唯一键 `model,modality,feature,eid,dose_sd`。基线配对键为同模型/分子/EID 的 `dose_sd=0`。

| 字段 | 定义 |
|---|---|
| model, model_role, eid, modality, feature, candidate_role, cohort_type, strictly_untrained | 同 cohort |
| baseline_value, dose_sd, target_value, is_baseline | 基线、SD 位移、理论目标、是否 0 SD |
| support_lower, support_upper, support_n | 训练观测分位数下/上界、不同训练 EID 观测数 |
| out_of_support | 目标越界；候选/支持缺失则为空 |
| executable | 后续联合执行许可：主要队列资格、分子已观测、有训练支持且非零剂量未越界 |
| nonexecution_reason | `candidate_missing`、`out_of_support`、`no_observed_training_support`、`exploratory_requires_separate_executor` 等分号分隔原因；可执行时 `none` |

### experiment_config.json

| 字段 | 内容 |
|---|---|
| schema_version, phase, generated_at_utc | 版本、4.4.1、UTC 生成时间 |
| git_commit, git_worktree_status, code_files | commit、未提交文件状态及相关源代码 SHA256；避免仅 commit 无法区分工作树 |
| input_files, input_directories | 全部必需输入的绝对路径、SHA256，以及 4.2/4.3/数据目录 |
| models | 有序模型注册表：label、role、directory、weights_path/weights_sha256、split_seed/model_seed、best_tau、完整原始 used_config |
| outcomes | 配置顺序 continuous/binary/ordered；report_groups 中肝功能/其他连续/三项 hepatic failure；标签解释声明 |
| selection, cohort_rules, dose_design, random_seed | 全部候选规则与数量、患者规则、训练支持与 0 SD 例外、剂量与随机种子 |
| outputs | 四份 CSV 的绝对路径、SHA256、行数与有序列名 |
| phase2_requirements, interpretation | 第二阶段前置要求与解释范围 |

### preflight_report.json

`code_version` 在成功和失败报告中均保存 Git commit、工作树状态与相关源代码哈希。

`schema_version,status,generated_at_utc,dry_run,errors,warnings,missing_files` 描述检查状态。`input_files` 为实际输入哈希；`candidate_counts` 按模态/角色计数。每个 `models[label]` 记录审计状态或错误，以及通过检查模型的总体/严格未训练/可用/主要队列人数、`per_candidate` 的合格人数、缺失率、非零已观测主要队列剂量越界率（只以有支持状态的行作分母）、可执行非零行数，以及 `training_support` 明细。分母为空的比率为 null。失败诊断可能有 `selection_preview`；dry-run 成功不保存预览。成功正式运行另有 `outputs,experiment_config_sha256`。

## 第二阶段读取约定与现有阻塞

4.4.2 必须先读取 `preflight_report.json`，要求 `status=ready`；核对 `experiment_config_sha256`、所有 CSV 和输入/权重 SHA256，核对 schema，再按 plan 的 `executable` 与基线配对键执行。不得重新选择候选、患者、剂量；不得把探索队列合并进主要队列。

当前代码 `reparameterize()` 在 eval 下仍采样。第二阶段必须先明确共享/私有潜变量用均值或受控配对随机数、SCM gate 的确定性、outcome exogenous 噪声策略和 best_tau。此阶段不重构前向过程。仅检查权重文件存在、非空和哈希，不反序列化；历史配置可能保留 `hidden_dims.outcome/outcome_hidden_dims`，当前构造器不使用它们，报告警告并保留原配置。第二阶段仍需严格核对 checkpoint 键和形状，不能静默过滤旧 outcome 模块。

原预处理以划分前全部可用样本计算 mean/std；此处沿用已有输入 SD，不改变训练资产。训练分位数仅来自对应训练 EID，因此不会新增验证分位数泄漏，但不能据此宣称原输入标准化或现有验证流程已完全无泄漏。三个 hepatic failure 标签不自动视为 EASL-CLIF 确认的 ACLF。

## 测试

```powershell
conda run --no-capture-output -n casualvae_01 python -m unittest discover -s tests -p test_prepare_virtual_perturbation.py -v
python -m pytest tests/test_prepare_virtual_perturbation.py -q
python -m pytest -q
```

测试仅在临时目录构造合成数据与不可反序列化的模型占位字节，不读取真实患者，不构造真实实验产物。
