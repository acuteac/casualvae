"""
生成完整特征配置文件（基于原始增强数据）
"""

import json
import os


def generate_full_config():
    # 读取原始配置（用于 anchor 和 phenotype）
    original_config_path = "d:/working space/casualmodule/aclf_partial_multimodal_bundle/aclf_partial_multimodal_config.json"
    with open(original_config_path, 'r', encoding='utf-8') as f:
        original_config = json.load(f)

    # 读取完整特征列名（由 prepare_full_augmented_structured.py 生成）
    full_feature_path = "d:/working space/casualmodule/casualvae/outputs/structured_full/full_feature_columns.json"

    if not os.path.exists(full_feature_path):
        print(f"错误：未找到 {full_feature_path}")
        print("请先运行 prepare_full_augmented_structured.py 生成完整特征列名")
        return

    with open(full_feature_path, 'r', encoding='utf-8') as f:
        full_features = json.load(f)

    # 构建新配置
    new_config = {
        "anchor_cols": original_config["anchor_cols"],
        "phenotype_continuous_cols": original_config["phenotype_continuous_cols"],
        "phenotype_binary_cols": original_config.get("phenotype_binary_cols", []),
        "protein_cols": full_features["protein_cols"],
        "metabolite_cols": full_features["metabolite_cols"],
        "protein_availability_threshold": 0.5,
        "metabolite_availability_threshold": 0.5,
        "shared_dims": {
            "protein": 6,
            "metabolite": 6,
            "outcome": 6
        },
        "private_dims": {
            "protein": 3,
            "metabolite": 3
        },
        "hidden_dims": {
            "protein": [512, 256, 128],
            "metabolite": [256, 128, 64],
            "outcome": [64, 32]
        },
        "outcome_hidden_dims": [64, 32],
        "notes": {
            "data_source": "full augmented data (all protein and metabolite features)",
            "n_protein_features": full_features["n_protein_features"],
            "n_metabolite_features": full_features["n_metabolite_features"],
            "storm_liver_proxy_score": "mean of signed standardized clinical markers; higher means worse inflammation/liver injury",
            "binary_labels": "auxiliary hepatic failure subtype heads from labels.csv",
            "protein_selection": "all available proteins from augmented data",
            "metabolite_selection": "all available metabolites from augmented data",
            "outcome_in_scm": "outcome variables are integrated into the SCM as downstream nodes"
        }
    }

    # 保存新配置
    output_path = "d:/working space/casualmodule/casualvae/aclf_full_multimodal_config.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_config, f, ensure_ascii=False, indent=2)

    print(f"配置文件已生成: {output_path}")
    print(f"  蛋白质特征数: {full_features['n_protein_features']}")
    print(f"  代谢物特征数: {full_features['n_metabolite_features']}")
    print(f"  锚点变量数: {len(new_config['anchor_cols'])}")
    print(f"  连续型结局数: {len(new_config['phenotype_continuous_cols'])}")
    print(f"  二分类结局数: {len(new_config['phenotype_binary_cols'])}")


if __name__ == "__main__":
    generate_full_config()
