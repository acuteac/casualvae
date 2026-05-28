"""
从增强数据生成模型需要的 4 个结构化 CSV 文件（使用全部蛋白质和代谢物特征）：
- anchors_structured.csv（保持不变）
- phenotype_structured.csv（保持不变）
- protein_full_structured.csv（包含所有蛋白质列）
- metabolite_full_structured.csv（包含所有代谢物列）
"""

import os
import pandas as pd
import json


def to_jsonable_number(value):
    if pd.isna(value):
        return None
    return float(value)


def is_artifact_column(col: str) -> bool:
    return col == "index" or col.startswith("Unnamed")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "eid" and not is_artifact_column(c)]


def numeric_columns(df: pd.DataFrame, cols: list[str]) -> tuple[list[str], list[str]]:
    numeric = []
    non_numeric = []
    for c in cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
        else:
            non_numeric.append(c)
    return numeric, non_numeric


def informative_numeric_columns(df: pd.DataFrame, cols: list[str]) -> tuple[list[str], dict]:
    kept = []
    dropped_all_missing = []
    dropped_constant = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        non_missing = s.dropna()
        if non_missing.empty:
            dropped_all_missing.append(c)
        elif non_missing.nunique() <= 1:
            dropped_constant.append(c)
        else:
            kept.append(c)
    return kept, {
        "dropped_all_missing": dropped_all_missing,
        "dropped_constant": dropped_constant,
    }


def standardize_frame(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    params = {}
    for c in cols:
        s = pd.to_numeric(out[c], errors="coerce")
        mean = s.mean(skipna=True)
        std = s.std(skipna=True, ddof=0)
        if pd.isna(std) or std == 0:
            std = 1.0
        out[c] = (s - mean) / std
        params[c] = {"mean": to_jsonable_number(mean), "std": to_jsonable_number(std)}
    return out, params


def availability_counts(protein_df: pd.DataFrame, metabolite_df: pd.DataFrame, protein_cols: list[str], metabolite_cols: list[str]) -> dict:
    protein_avail = protein_df[protein_cols].notna().mean(axis=1) >= 0.5 if protein_cols else pd.Series(False, index=protein_df.index)
    metabolite_avail = metabolite_df[metabolite_cols].notna().mean(axis=1) >= 0.5 if metabolite_cols else pd.Series(False, index=metabolite_df.index)
    protein_ids = set(protein_df.loc[protein_avail, "eid"])
    metabolite_ids = set(metabolite_df.loc[metabolite_avail, "eid"])
    return {
        "protein_available_ge_0_5": int(len(protein_ids)),
        "metabolite_available_ge_0_5": int(len(metabolite_ids)),
        "paired_available_ge_0_5": int(len(protein_ids & metabolite_ids)),
    }


def prepare_full_structured_data(
    augmented_dir: str,
    config_path: str,
    output_dir: str,
):
    """
    从增强数据生成结构化输入文件（使用全部特征）

    Args:
        augmented_dir: 增强数据目录
        config_path: 配置文件路径（仅用于读取 anchor 和 phenotype 列）
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 读取配置（仅用于 anchor 和 phenotype）
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    anchor_cols = config['anchor_cols']
    phenotype_cont_cols = config['phenotype_continuous_cols']
    phenotype_bin_cols = config.get('phenotype_binary_cols', [])
    preprocessing_params = {"anchor": {}, "phenotype_continuous": {}, "protein": {}, "metabolite": {}}
    quality_summary = {"dropped_columns": {}, "availability": {}}

    # 读取增强数据
    ukb_df = pd.read_csv(os.path.join(augmented_dir, 'UKB_data_liver_failure.csv'), low_memory=False)
    protein_df = pd.read_csv(os.path.join(augmented_dir, 'Protein_all_instance_0_liver_failure.csv'), low_memory=False)
    nmr_df = pd.read_csv(os.path.join(augmented_dir, 'NMR_liver_failure.csv'), low_memory=False)
    labels_df = pd.read_csv(os.path.join(augmented_dir, 'liver_failure_case_labels.csv'), low_memory=False)

    print(f"读取数据:")
    print(f"  UKB: {len(ukb_df)} 行, {len(ukb_df.columns)} 列")
    print(f"  Protein: {len(protein_df)} 行, {len(protein_df.columns)} 列")
    print(f"  NMR: {len(nmr_df)} 行, {len(nmr_df.columns)} 列")
    print(f"  Labels: {len(labels_df)} 行, {len(labels_df.columns)} 列")

    # 1. 生成 anchors_structured.csv（保持不变）
    anchor_candidates = [c for c in anchor_cols if c in ukb_df.columns]
    anchor_numeric_cols, anchor_non_numeric = numeric_columns(ukb_df, anchor_candidates)
    anchors_structured = ukb_df[['eid'] + anchor_numeric_cols].copy()
    anchors_structured, preprocessing_params["anchor"] = standardize_frame(anchors_structured, anchor_numeric_cols)
    quality_summary["dropped_columns"]["anchor_non_numeric"] = anchor_non_numeric
    anchors_output = os.path.join(output_dir, 'anchors_structured.csv')
    anchors_structured.to_csv(anchors_output, index=False)
    print(f"\n生成 anchors_structured.csv: {len(anchors_structured)} 行, {len(anchors_structured.columns)} 列")

    # 2. 生成 phenotype_structured.csv（保持不变）
    phenotype_merged = ukb_df[['eid']].merge(labels_df, on='eid', how='left')

    phenotype_cols_available = ['eid']
    phenotype_cont_available = []
    for c in phenotype_cont_cols:
        if c in ukb_df.columns:
            phenotype_merged[c] = ukb_df[c]
            phenotype_cols_available.append(c)
            phenotype_cont_available.append(c)
        elif c in labels_df.columns:
            phenotype_cols_available.append(c)
            phenotype_cont_available.append(c)

    phenotype_bin_available = []
    for c in phenotype_bin_cols:
        if c in labels_df.columns:
            phenotype_cols_available.append(c)
            phenotype_bin_available.append(c)

    phenotype_structured = phenotype_merged[phenotype_cols_available].copy()
    phenotype_structured, preprocessing_params["phenotype_continuous"] = standardize_frame(
        phenotype_structured, phenotype_cont_available
    )
    for c in phenotype_bin_available:
        phenotype_structured[c] = phenotype_structured[c].fillna(False).astype(float)
    phenotype_output = os.path.join(output_dir, 'phenotype_structured.csv')
    phenotype_structured.to_csv(phenotype_output, index=False)
    print(f"生成 phenotype_structured.csv: {len(phenotype_structured)} 行, {len(phenotype_structured.columns)} 列")

    # 3. 生成 protein_full_structured.csv（包含所有蛋白质列）
    protein_cols_only, protein_drop_info = informative_numeric_columns(protein_df, feature_columns(protein_df))
    metabolite_cols_only, metabolite_drop_info = informative_numeric_columns(nmr_df, feature_columns(nmr_df))
    quality_summary["dropped_columns"]["protein"] = protein_drop_info
    quality_summary["dropped_columns"]["metabolite"] = metabolite_drop_info

    protein_full_structured = protein_df[["eid"] + protein_cols_only].copy()
    protein_full_structured, preprocessing_params["protein"] = standardize_frame(protein_full_structured, protein_cols_only)
    protein_output = os.path.join(output_dir, 'protein_full_structured.csv')
    protein_full_structured.to_csv(protein_output, index=False)
    print(f"生成 protein_full_structured.csv: {len(protein_full_structured)} 行, {len(protein_full_structured.columns)} 列")

    # 计算蛋白质特征的缺失率统计
    protein_missing_rates = protein_df[protein_cols_only].isna().mean().sort_values(ascending=False)
    print(f"  蛋白质特征数: {len(protein_cols_only)}")
    print(f"  平均缺失率: {protein_missing_rates.mean():.2%}")
    print(f"  缺失率中位数: {protein_missing_rates.median():.2%}")
    print(f"  缺失率范围: [{protein_missing_rates.min():.2%}, {protein_missing_rates.max():.2%}]")

    # 4. 生成 metabolite_full_structured.csv（包含所有代谢物列）
    metabolite_full_structured = nmr_df[["eid"] + metabolite_cols_only].copy()
    metabolite_full_structured, preprocessing_params["metabolite"] = standardize_frame(
        metabolite_full_structured, metabolite_cols_only
    )
    metabolite_output = os.path.join(output_dir, 'metabolite_full_structured.csv')
    metabolite_full_structured.to_csv(metabolite_output, index=False)
    print(f"生成 metabolite_full_structured.csv: {len(metabolite_full_structured)} 行, {len(metabolite_full_structured.columns)} 列")

    # 计算代谢物特征的缺失率统计
    metabolite_missing_rates = nmr_df[metabolite_cols_only].isna().mean().sort_values(ascending=False)
    print(f"  代谢物特征数: {len(metabolite_cols_only)}")
    print(f"  平均缺失率: {metabolite_missing_rates.mean():.2%}")
    print(f"  缺失率中位数: {metabolite_missing_rates.median():.2%}")
    print(f"  缺失率范围: [{metabolite_missing_rates.min():.2%}, {metabolite_missing_rates.max():.2%}]")

    # 验证
    print(f"\n验证:")
    print(f"  anchors eid 唯一值: {anchors_structured['eid'].nunique()}")
    print(f"  phenotype eid 唯一值: {phenotype_structured['eid'].nunique()}")
    print(f"  protein eid 唯一值: {protein_full_structured['eid'].nunique()}")
    print(f"  metabolite eid 唯一值: {metabolite_full_structured['eid'].nunique()}")

    quality_summary["availability"] = availability_counts(
        protein_full_structured, metabolite_full_structured, protein_cols_only, metabolite_cols_only
    )
    quality_summary["feature_counts"] = {
        "anchor": len(anchor_numeric_cols),
        "phenotype_continuous": len(phenotype_cont_available),
        "phenotype_binary": len(phenotype_bin_available),
        "protein": len(protein_cols_only),
        "metabolite": len(metabolite_cols_only),
    }
    quality_summary_path = os.path.join(output_dir, 'data_quality_summary.json')
    with open(quality_summary_path, 'w', encoding='utf-8') as f:
        json.dump(quality_summary, f, ensure_ascii=False, indent=2)

    preprocessing_params_path = os.path.join(output_dir, 'preprocessing_params.json')
    with open(preprocessing_params_path, 'w', encoding='utf-8') as f:
        json.dump(preprocessing_params, f, ensure_ascii=False, indent=2)

    # 保存列名列表到 JSON（用于生成配置文件）
    column_info = {
        "protein_cols": protein_cols_only,
        "metabolite_cols": metabolite_cols_only,
        "n_protein_features": len(protein_cols_only),
        "n_metabolite_features": len(metabolite_cols_only),
    }
    column_info_path = os.path.join(output_dir, 'full_feature_columns.json')
    with open(column_info_path, 'w', encoding='utf-8') as f:
        json.dump(column_info, f, ensure_ascii=False, indent=2)
    print(f"\n列名信息已保存到: {column_info_path}")

    print(f"\n所有文件已保存到: {output_dir}")


def main():
    augmented_dir = "d:/working space/casualmodule/casualvae/outputs/augmented"
    config_path = "d:/working space/casualmodule/aclf_partial_multimodal_bundle/aclf_partial_multimodal_config.json"
    output_dir = "d:/working space/casualmodule/casualvae/outputs/structured_full"

    prepare_full_structured_data(augmented_dir, config_path, output_dir)


if __name__ == "__main__":
    main()
