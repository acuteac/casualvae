"""
数据预处理和增强脚本 - 修复版
只对连续型 float64 列添加噪声，跳过分类/整数/日期列
"""

import os
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_continuous_cols(df: pd.DataFrame, exclude_cols: list) -> list:
    """
    识别真正的连续变量列：
    - 必须是 float64
    - 唯一值数量 > 20（排除编码为float的分类变量）
    - 不在排除列表中
    - 排除明显的分类/年份/编码列
    """
    # 扩展排除列表：年份、中心编码、日期相关
    categorical_patterns = [
        'year', 'center', 'date', 'age', 'code', 'type', 'status',
        'ethnic', 'sex', 'qualification', 'employment', 'group',
        'assess_center', 'birth_year', 'recruit_age'
    ]

    continuous = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if df[col].dtype != np.float64:
            continue

        # 检查列名是否包含分类模式
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in categorical_patterns):
            continue

        n_unique = df[col].dropna().nunique()
        n_total = df[col].dropna().shape[0]

        # 如果唯一值太少（< 20）或唯一值比例太低（< 5%），跳过
        if n_unique <= 20:
            continue
        if n_total > 0 and n_unique / n_total < 0.05:
            continue

        # 检查是否所有值都是整数（可能是编码）
        non_null = df[col].dropna()
        if len(non_null) > 0 and (non_null == non_null.astype(int)).all():
            continue

        continuous.append(col)
    return continuous


def add_gaussian_noise(df: pd.DataFrame, continuous_cols: list, noise_level: float = 0.05, seed: int = 42) -> pd.DataFrame:
    """只对连续列添加高斯噪声"""
    rng = np.random.default_rng(seed)
    aug = df.copy()
    for col in continuous_cols:
        valid = ~aug[col].isna()
        if valid.sum() == 0:
            continue
        std = aug.loc[valid, col].std()
        if std == 0 or np.isnan(std):
            continue
        noise = rng.normal(0, std * noise_level, valid.sum())
        aug.loc[valid, col] = aug.loc[valid, col] + noise
    return aug


def augment_file(input_path: str, output_path: str, seed: int = 42):
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  原始: {len(df)} 行, {len(df.columns)} 列")

    exclude_cols = ['eid']
    if 'Unnamed: 0' in df.columns:
        exclude_cols.append('Unnamed: 0')

    continuous_cols = get_continuous_cols(df, exclude_cols)
    print(f"  连续变量列数: {len(continuous_cols)}")

    aug_df = add_gaussian_noise(df, continuous_cols, noise_level=0.05, seed=seed)

    # 为增强样本生成新 eid（原始最大值之后）
    max_eid = df['eid'].max()
    aug_df['eid'] = df['eid'] + max_eid

    combined = pd.concat([df, aug_df], ignore_index=True)
    print(f"  增强后: {len(combined)} 行")
    combined.to_csv(output_path, index=False)


def main():
    input_dir = str(PROJECT_ROOT / "data" / "raw" / "extract")
    output_dir = str(PROJECT_ROOT / "data" / "augmented")
    os.makedirs(output_dir, exist_ok=True)

    files = [
        'UKB_data_liver_failure.csv',
        'Protein_all_instance_0_liver_failure.csv',
        'NMR_liver_failure.csv',
        'liver_failure_case_labels.csv',
        'ukb672226_liver_failure_diagnosis_subset.csv',
    ]

    for i, fname in enumerate(files):
        src = os.path.join(input_dir, fname)
        dst = os.path.join(output_dir, fname)
        if not os.path.exists(src):
            print(f"跳过（不存在）: {fname}")
            continue
        print(f"\n处理: {fname}")
        augment_file(src, dst, seed=42 + i)

    # 更新摘要
    summary_src = os.path.join(input_dir, 'extract_summary.csv')
    summary_dst = os.path.join(output_dir, 'extract_summary.csv')
    if os.path.exists(summary_src):
        s = pd.read_csv(summary_src)
        if 'matched_rows' in s.columns:
            s['matched_rows'] = s['matched_rows'] * 2
        s.to_csv(summary_dst, index=False)

    print("\n完成！")


if __name__ == "__main__":
    main()
