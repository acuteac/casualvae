# 4.3 总结构传播效应可视化说明

本目录是 **4.3 总结构传播效应分析** 的可视化结果。分析单位始终是：

```text
Protein / Metabolite → Outcome
```

总效应先在 latent SCM 中计算，再投影到 feature→Outcome：

```text
T_A = (I - A)^-1 - I
G_T = D × T_A × D^T
indirect_effect = total_effect - direct_effect
```

本目录只负责结果展示。可供程序读取的正式 CSV/JSON 位于：

```text
artifacts/analysis/total_effect_propagation/
```

## 1. 如何查看

首先打开 [`index.html`](index.html)。它是所有 Outcome 页面的入口。

- 页面是静态 HTML，不需要启动 Web 服务。
- 表格内容可以离线查看。
- 图表使用 Plotly CDN；如果计算机完全离线，表格仍会显示，但散点图和热图可能无法加载。
- 代谢物统一显示为 `真实名称 [原始编码]`，例如 `Acetone [p23477_i0]`。

## 2. `index.html` 的含义

入口页包含两部分。

### 总传播效应排名的跨模型稳定性

热图比较 Primary 与另外三个模型的 Total ranking Spearman 相关性。

- 每一行是一个 `modality × Outcome`。
- 每一列是一个比较模型。
- 数值越接近 1，说明整体 Total 排名越相似。
- Spearman 只描述完整排名的相似性；Top20/Top50/Top100 的精确重叠情况应查看正式稳定性 CSV。

### 多 Outcome 总效应候选

该表统计同一特征在多少个 Outcome 中重复进入 Total Top20、Top50，并列出对应 Outcome。

它适合寻找跨 Outcome 重复信号，但不能单独证明某个特征是 biomarker。

## 3. Outcome 页面

每个 `outcome_*.html` 对应一个 Outcome：

| 文件 | Outcome 含义 |
|---|---|
| `outcome_CRP.html` | C-reactive protein，C 反应蛋白 |
| `outcome_WBC.html` | White blood cell count，白细胞计数 |
| `outcome_NEUT.html` | Neutrophil count，中性粒细胞计数 |
| `outcome_NEUT_P.html` | Neutrophil percentage，中性粒细胞比例 |
| `outcome_PLT.html` | Platelet count，血小板计数 |
| `outcome_ALT.html` | Alanine aminotransferase，丙氨酸氨基转移酶 |
| `outcome_AST.html` | Aspartate aminotransferase，天冬氨酸氨基转移酶 |
| `outcome_ALP.html` | Alkaline phosphatase，碱性磷酸酶 |
| `outcome_GGT.html` | Gamma-glutamyl transferase，γ-谷氨酰转移酶 |
| `outcome_TBil.html` | Total bilirubin，总胆红素 |
| `outcome_DBil.html` | Direct bilirubin，直接胆红素 |
| `outcome_ALB.html` | Albumin，白蛋白 |
| `outcome_Urea.html` | Urea，尿素 |
| `outcome_Crea.html` | Creatinine，肌酐 |
| `outcome_acute_or_subacute_hepatic_failure.html` | 急性或亚急性肝衰竭 |
| `outcome_chronic_hepatic_failure.html` | 慢性肝衰竭 |
| `outcome_hepatic_failure_unspecified.html` | 未特指肝衰竭 |

每个 Outcome 页面包含以下栏目。

### 蛋白质总效应 Top20

在该 Outcome 内，按 `abs(total_effect)` 从高到低列出前 20 个 Protein。

### 代谢物总效应 Top20

在该 Outcome 内，按 `abs(total_effect)` 从高到低列出前 20 个 Metabolite。特征列直接给出真实代谢物名称，并在方括号内保留原始编码。

### 直接效应排名与总效应排名

散点图中的每个点是一条 `feature → Outcome` 边：

- 横轴：Direct Outcome rank。
- 纵轴：Total Outcome rank。
- 两个轴均使用对数刻度。
- `rank_shift > 0` 表示网络传播后排名上升。
- `rank_shift < 0` 表示网络传播后排名下降。
- 悬浮在点上可以查看特征名称、Direct rank、Total rank 和 RankShift。

Protein 与 Metabolite 分别排名，不能把两种模态的 rank 直接混合比较。

### 间接放大候选

定义为：

```text
direct_outcome_rank > 100
total_outcome_rank <= 100
```

这表示直接效应没有进入 Top100，但经过 latent SCM 网络传播后，总效应进入了 Top100。

### 衰减候选

定义为：

```text
direct_outcome_rank <= 100
total_outcome_rank > 100
```

这表示直接效应原本较强，但间接传播可能削弱或抵消了该效应。

### 方向翻转候选

定义为：

```text
direct_effect × total_effect < 0
```

方向翻转表示完整网络传播后总效应符号发生改变。必须同时查看效应量和排名；非常接近 0 的尾部效应即使翻转，也不宜过度解释。

## 4. 表格字段说明

| 页面字段 | 含义 |
|---|---|
| 模态 | Protein 或 Metabolite |
| 特征 | Protein 名称，或 `代谢物真实名称 [原始编码]` |
| 直接效应 | 4.2 中的 `D × A × D^T` 直接结构投影效应 |
| 总效应 | 完整 latent SCM 传播后的 `D × [(I-A)^-1-I] × D^T` |
| 直接效应排名 | 在同一 `modality × Outcome` 内按 `abs(direct_effect)` 排名 |
| 总效应排名 | 在同一 `modality × Outcome` 内按 `abs(total_effect)` 排名 |
| 排名变化 | `direct_outcome_rank - total_outcome_rank` |
| 总效应同号支持（/3） | 另外三个模型中，与 Primary 的 total effect 同号的模型数 |
| Total Top20 支持（/4） | 四个模型中，该边进入相同 Outcome Total Top20 的模型数 |
| 间接放大支持（/4） | 四个模型中，该边同时满足 indirect-amplified 定义的模型数 |

## 5. 四个模型的角色

```text
Primary:
artifacts/multiseed/fixed_split/model_seed_456

Replication:
artifacts/multiseed/variable_split/seed_456
artifacts/multiseed/variable_split/seed_789

Sensitivity:
artifacts/multiseed/fixed_split/model_seed_789
```

正式候选以 Primary 为主，另外三个模型仅用于判断方向、排名和模式是否复现。四个模型的 A 和 effect 没有被平均。

## 6. 对应的正式数据文件

| 数据文件 | 内容 |
|---|---|
| `total_effect_primary.csv` | Primary 的全部 53,958 条 molecular→Outcome 边 |
| `total_effect_cross_model_comparison.csv` | 同一条边在四模型中的效应、排名与支持计数 |
| `total_effect_top_edges.csv` | 每个 Outcome×modality 的 Primary Total Top100 |
| `direct_vs_total_candidates.csv` | direct-and-total-high、indirect-amplified、attenuated 和 direction-reversal 候选 |
| `total_effect_stability_summary.csv` | Primary 与各比较模型的 Spearman 和 Top-K Jaccard |
| `multi_outcome_total_effect_summary.csv` | 每个特征跨 Outcome 重复进入 Total Top-K 的情况 |
| `total_effect_summary.json` | 模型角色、数值状态、维度、阈值和模式数量摘要 |

这些文件位于：

```text
artifacts/analysis/total_effect_propagation/
```

## 7. 解读注意事项

1. 排名是 Outcome-specific 且 modality-specific 的相对排名。
2. 不能只根据 RankShift 判断 biomarker；还要同时考虑效应量、符号和跨模型支持。
3. `total_sign_agreement_count` 的分母是 3；Top-K 和模式支持计数的分母是 4。
4. `direction_reversal` 是描述性标记，不等于稳健的生物学反向作用。
5. Multi-outcome 汇总是辅助结果，不与 4.2 或稳定性指标强行合成为单一评分。
6. 可视化中的数值来自正式 CSV；若需要筛选、统计或复核，应优先读取 CSV，而不是解析 HTML。

## 8. 重新生成

在项目根目录执行：

```powershell
python -m scripts.analysis.total_effect_propagation
```

该命令会重新生成正式 CSV/JSON，并覆盖更新本目录中的 `index.html` 和 17 个 Outcome 页面。
