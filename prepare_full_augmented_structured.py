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
    anchor_cols_available = ['eid'] + [c for c in anchor_cols if c in ukb_df.columns]
    anchors_structured = ukb_df[anchor_cols_available].copy()
    anchors_output = os.path.join(output_dir, 'anchors_structured.csv')
    anchors_structured.to_csv(anchors_output, index=False)
    print(f"\n生成 anchors_structured.csv: {len(anchors_structured)} 行, {len(anchors_structured.columns)} 列")

    # 2. 生成 phenotype_structured.csv（保持不变）
    phenotype_merged = ukb_df[['eid']].merge(labels_df, on='eid', how='left')

    phenotype_cols_available = ['eid']
    for c in phenotype_cont_cols:
        if c in ukb_df.columns:
            phenotype_merged[c] = ukb_df[c]
            phenotype_cols_available.append(c)
        elif c in labels_df.columns:
            phenotype_cols_available.append(c)

    for c in phenotype_bin_cols:
        if c in labels_df.columns:
            phenotype_cols_available.append(c)

    phenotype_structured = phenotype_merged[phenotype_cols_available].copy()
    phenotype_output = os.path.join(output_dir, 'phenotype_structured.csv')
    phenotype_structured.to_csv(phenotype_output, index=False)
    print(f"生成 phenotype_structured.csv: {len(phenotype_structured)} 行, {len(phenotype_structured.columns)} 列")

    # 3. 生成 protein_full_structured.csv（包含所有蛋白质列）
    protein_full_structured = protein_df.copy()
    protein_output = os.path.join(output_dir, 'protein_full_structured.csv')
    protein_full_structured.to_csv(protein_output, index=False)
    print(f"生成 protein_full_structured.csv: {len(protein_full_structured)} 行, {len(protein_full_structured.columns)} 列")

    # 计算蛋白质特征的缺失率统计
    protein_cols_only = [c for c in protein_df.columns if c != 'eid']
    protein_missing_rates = protein_df[protein_cols_only].isna().mean().sort_values(ascending=False)
    print(f"  蛋白质特征数: {len(protein_cols_only)}")
    print(f"  平均缺失率: {protein_missing_rates.mean():.2%}")
    print(f"  缺失率中位数: {protein_missing_rates.median():.2%}")
    print(f"  缺失率范围: [{protein_missing_rates.min():.2%}, {protein_missing_rates.max():.2%}]")

    # 4. 生成 metabolite_full_structured.csv（包含所有代谢物列）
    metabolite_full_structured = nmr_df.copy()
    metabolite_output = os.path.join(output_dir, 'metabolite_full_structured.csv')
    metabolite_full_structured.to_csv(metabolite_output, index=False)
    print(f"生成 metabolite_full_structured.csv: {len(metabolite_full_structured)} 行, {len(metabolite_full_structured.columns)} 列")

    # 计算代谢物特征的缺失率统计
    metabolite_cols_only = [c for c in nmr_df.columns if c != 'eid']
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

