"""
从增强数据生成模型需要的 4 个结构化 CSV 文件：
- anchors_structured.csv
- phenotype_structured.csv
- protein_structured.csv
- metabolite_structured.csv
"""

import os
import pandas as pd
import json


def prepare_structured_data(
    augmented_dir: str,
    config_path: str,
    output_dir: str,
):
    """
    从增强数据生成结构化输入文件

    Args:
        augmented_dir: 增强数据目录
        config_path: 配置文件路径
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 读取配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    anchor_cols = config['anchor_cols']
    phenotype_cont_cols = config['phenotype_continuous_cols']
    phenotype_bin_cols = config.get('phenotype_binary_cols', [])
    protein_cols = config['protein_cols']
    metabolite_cols = config['metabolite_cols']

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

    # 1. 生成 anchors_structured.csv
    anchor_cols_available = ['eid'] + [c for c in anchor_cols if c in ukb_df.columns]
    anchors_structured = ukb_df[anchor_cols_available].copy()
    anchors_output = os.path.join(output_dir, 'anchors_structured.csv')
    anchors_structured.to_csv(anchors_output, index=False)
    print(f"\n生成 anchors_structured.csv: {len(anchors_structured)} 行, {len(anchors_structured.columns)} 列")

    # 2. 生成 phenotype_structured.csv
    # 合并 UKB 和 labels 数据
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

    # 3. 生成 protein_structured.csv
    protein_cols_available = ['eid'] + [c for c in protein_cols if c in protein_df.columns]
    protein_structured = protein_df[protein_cols_available].copy()
    protein_output = os.path.join(output_dir, 'protein_structured.csv')
    protein_structured.to_csv(protein_output, index=False)
    print(f"生成 protein_structured.csv: {len(protein_structured)} 行, {len(protein_structured.columns)} 列")

    # 4. 生成 metabolite_structured.csv
    metabolite_cols_available = ['eid'] + [c for c in metabolite_cols if c in nmr_df.columns]
    metabolite_structured = nmr_df[metabolite_cols_available].copy()
    metabolite_output = os.path.join(output_dir, 'metabolite_structured.csv')
    metabolite_structured.to_csv(metabolite_output, index=False)
    print(f"生成 metabolite_structured.csv: {len(metabolite_structured)} 行, {len(metabolite_structured.columns)} 列")

    # 验证
    print(f"\n验证:")
    print(f"  anchors eid 唯一值: {anchors_structured['eid'].nunique()}")
    print(f"  phenotype eid 唯一值: {phenotype_structured['eid'].nunique()}")
    print(f"  protein eid 唯一值: {protein_structured['eid'].nunique()}")
    print(f"  metabolite eid 唯一值: {metabolite_structured['eid'].nunique()}")

    print(f"\n所有文件已保存到: {output_dir}")


def main():
    augmented_dir = "D:/working space/casualmodule/casualvae/outputs/augmented"
    config_path = "D:/working space/casualmodule/aclf_partial_multimodal_bundle/aclf_partial_multimodal_config.json"
    output_dir = "D:/working space/casualmodule/casualvae/outputs/structured"

    prepare_structured_data(augmented_dir, config_path, output_dir)


if __name__ == "__main__":
    main()
