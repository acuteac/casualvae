import argparse
import itertools
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_MODEL_SEEDS = (42, 123, 456, 789, 2026)
DEFAULT_TOP_KS = (20, 50, 100, 200)
EDGE_ID_COLUMNS = ["modality", "source", "target"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXED_SPLIT_DIR = PROJECT_ROOT / "artifacts" / "multiseed" / "fixed_split"
DEFAULT_VARIABLE_SPLIT_DIR = PROJECT_ROOT / "artifacts" / "multiseed" / "variable_split"
DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "artifacts" / "analysis" / "multiseed"


def _find_column(frame: pd.DataFrame, candidates: Sequence[str], label: str) -> str:
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    raise ValueError(
        f"training_history.csv has no {label} column; available columns: "
        f"{list(frame.columns)}"
    )


def extract_training_summary(
    run_dir: Path,
    fallback_model_seed: int,
    fallback_split_seed: Optional[int] = None,
) -> Dict[str, object]:
    run_dir = Path(run_dir)
    with (run_dir / "run_summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    history = pd.read_csv(run_dir / "training_history.csv")
    if history.empty:
        raise ValueError(f"Empty training history: {run_dir / 'training_history.csv'}")

    stage_column = next((column for column in history.columns if column.lower() == "stage"), None)
    joint_history = history
    if stage_column is not None:
        joint_history = history.loc[
            history[stage_column].astype(str).str.lower().eq("joint")
        ].copy()
    if joint_history.empty:
        raise ValueError(f"No joint-stage rows in {run_dir / 'training_history.csv'}")

    epoch_column = _find_column(joint_history, ("epoch",), "epoch")
    val_loss_column = _find_column(
        joint_history,
        ("val_loss", "validation_loss", "valid_loss", "val_total_loss"),
        "validation loss",
    )
    cont_column = _find_column(
        joint_history,
        ("val_outcome_cont", "val_outcome_cont_loss", "outcome_cont_val_loss"),
        "continuous-outcome validation loss",
    )
    bin_column = _find_column(
        joint_history,
        ("val_outcome_bin", "val_outcome_bin_loss", "outcome_bin_val_loss"),
        "binary-outcome validation loss",
    )

    best_epoch = summary.get("best_epoch")
    if best_epoch is None or not np.isfinite(float(best_epoch)):
        finite = pd.to_numeric(joint_history[val_loss_column], errors="coerce")
        if not finite.notna().any():
            raise ValueError(f"No finite validation loss in {run_dir / 'training_history.csv'}")
        best_index = finite.idxmin()
        best_epoch = int(joint_history.loc[best_index, epoch_column])

    epoch_values = pd.to_numeric(joint_history[epoch_column], errors="coerce")
    best_rows = joint_history.loc[epoch_values.eq(float(best_epoch))]
    if best_rows.empty:
        raise ValueError(
            f"best_epoch={best_epoch} is absent from joint-stage history in {run_dir}"
        )
    best_row = best_rows.iloc[-1]

    values = {
        "best_val_loss": pd.to_numeric(pd.Series([best_row[val_loss_column]]), errors="coerce").iloc[0],
        "outcome_cont_val_loss": pd.to_numeric(pd.Series([best_row[cont_column]]), errors="coerce").iloc[0],
        "outcome_bin_val_loss": pd.to_numeric(pd.Series([best_row[bin_column]]), errors="coerce").iloc[0],
    }
    invalid = [name for name, value in values.items() if not np.isfinite(value)]
    if invalid:
        raise ValueError(f"Non-finite best-epoch metrics {invalid} in {run_dir}")

    model_seed = int(summary.get("model_seed", fallback_model_seed))
    split_seed_value = summary.get("split_seed", fallback_split_seed)
    return {
        "seed": model_seed,
        "model_seed": model_seed,
        "split_seed": int(split_seed_value) if split_seed_value is not None else model_seed,
        "best_epoch": int(best_epoch),
        "best_tau": float(summary.get("best_tau", np.nan)),
        **{name: float(value) for name, value in values.items()},
    }


def _robust_outlier_mask(values: pd.Series, threshold: float = 3.5) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    mask = pd.Series(False, index=values.index)
    if len(finite) < 3:
        return mask
    median = float(finite.median())
    deviations = (finite - median).abs()
    mad = float(deviations.median())
    scale_floor = np.finfo(float).eps * max(1.0, abs(median))
    if mad > scale_floor:
        mask.loc[finite.index] = (0.67448975 * deviations / mad) > threshold
        return mask

    q1, q3 = finite.quantile([0.25, 0.75])
    iqr = float(q3 - q1)
    if iqr > scale_floor:
        lower = float(q1 - 1.5 * iqr)
        upper = float(q3 + 1.5 * iqr)
        mask.loc[finite.index] = (finite < lower) | (finite > upper)
    return mask


def flag_training_outliers(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    reasons: Dict[int, List[str]] = {index: [] for index in result.index}
    for column in (
        "best_epoch",
        "best_val_loss",
        "outcome_cont_val_loss",
        "outcome_bin_val_loss",
    ):
        mask = _robust_outlier_mask(result[column])
        for index in result.index[mask]:
            reasons[index].append(column)

    epoch_values = pd.to_numeric(result["best_epoch"], errors="coerce")
    epoch_median = float(epoch_values.median())
    if np.isfinite(epoch_median) and epoch_median > 0:
        very_early = epoch_values < (0.5 * epoch_median)
        for index in result.index[very_early]:
            if "best_epoch" not in reasons[index]:
                reasons[index].append("best_epoch")

    result["is_training_outlier"] = [bool(reasons[index]) for index in result.index]
    result["training_outlier_reasons"] = [";".join(reasons[index]) for index in result.index]
    return result


def load_edge_file(path: Path, modality: str, model_seed: int) -> pd.DataFrame:
    path = Path(path)
    raw = pd.read_csv(path)
    required_long = {"source", "target", "edge_weight"}
    if required_long.issubset(raw.columns):
        edges = raw.loc[:, ["source", "target", "edge_weight"]].copy()
    else:
        wide = pd.read_csv(path, index_col=0)
        wide.index.name = "target"
        edges = wide.reset_index().melt(
            id_vars="target", var_name="source", value_name="edge_weight"
        )
    edges["edge_weight"] = pd.to_numeric(edges["edge_weight"], errors="coerce")
    if edges["edge_weight"].isna().any():
        raise ValueError(f"Non-numeric edge weights in {path}")
    edges["source"] = edges["source"].astype(str)
    edges["target"] = edges["target"].astype(str)
    edges["model_seed"] = int(model_seed)
    edges["modality"] = modality
    return edges[["model_seed", "modality", "source", "target", "edge_weight"]]


def add_edge_ranks(edges: pd.DataFrame) -> pd.DataFrame:
    ranked = edges.copy().reset_index(drop=True)
    ranked["abs_edge_weight"] = ranked["edge_weight"].abs()
    ranked["rank"] = np.nan
    ranked["outcome_rank"] = np.nan

    for _, group in ranked.groupby(["model_seed", "modality"], sort=True):
        order = group.sort_values(
            ["abs_edge_weight", "target", "source"],
            ascending=[False, True, True],
            kind="mergesort",
        ).index
        ranked.loc[order, "rank"] = np.arange(1, len(order) + 1, dtype=float)

    for _, group in ranked.groupby(["model_seed", "modality", "target"], sort=True):
        order = group.sort_values(
            ["abs_edge_weight", "source"],
            ascending=[False, True],
            kind="mergesort",
        ).index
        ranked.loc[order, "outcome_rank"] = np.arange(1, len(order) + 1, dtype=float)

    ranked["rank"] = ranked["rank"].astype(int)
    ranked["outcome_rank"] = ranked["outcome_rank"].astype(int)
    return ranked


def aggregate_edge_statistics(
    ranked_edges: pd.DataFrame,
    model_seeds: Sequence[int],
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
    cv_absolute_floor: float = 1e-8,
    cv_relative_floor: float = 0.1,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    seed_count = len(model_seeds)
    for keys, group in ranked_edges.groupby(EDGE_ID_COLUMNS, sort=True):
        weights = group["edge_weight"].to_numpy(dtype=float)
        mean_effect = float(np.mean(weights))
        mean_abs_effect = float(np.mean(np.abs(weights)))
        std_effect = float(np.std(weights, ddof=0))
        cv_denominator_valid = abs(mean_effect) >= max(
            cv_absolute_floor,
            cv_relative_floor * mean_abs_effect,
        )
        positive_count = int(np.sum(weights > 0))
        negative_count = int(np.sum(weights < 0))
        observed_count = int(len(weights))
        row: Dict[str, object] = {
            "modality": keys[0],
            "source": keys[1],
            "target": keys[2],
            "mean_effect": mean_effect,
            "median_effect": float(np.median(weights)),
            "mean_abs_effect": mean_abs_effect,
            "std_effect": std_effect,
            "effect_cv": std_effect / abs(mean_effect) if cv_denominator_valid else np.nan,
            "effect_cv_denominator_valid": bool(cv_denominator_valid),
            "sign_consistency": max(positive_count, negative_count) / observed_count,
            "positive_seed_count": positive_count,
            "negative_seed_count": negative_count,
            "observed_seed_count": observed_count,
            "mean_rank": float(group["rank"].mean()),
            "median_rank": float(group["rank"].median()),
        }
        for top_k in top_ks:
            count = int((group["rank"] <= top_k).sum())
            row[f"top{top_k}_count"] = count
            row[f"top{top_k}_frequency"] = count / seed_count if seed_count else np.nan
        by_seed = group.set_index("model_seed")["edge_weight"]
        for model_seed in model_seeds:
            row[f"model_seed_{model_seed}"] = (
                float(by_seed.loc[model_seed]) if model_seed in by_seed.index else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def compute_pairwise_topk_stability(
    ranked_edges: pd.DataFrame,
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    for modality, modality_edges in ranked_edges.groupby("modality", sort=True):
        seeds = sorted(modality_edges["model_seed"].unique())
        for seed_a, seed_b in itertools.combinations(seeds, 2):
            for top_k in top_ks:
                set_a = set(
                    modality_edges.loc[
                        (modality_edges["model_seed"] == seed_a)
                        & (modality_edges["rank"] <= top_k),
                        ["source", "target"],
                    ].itertuples(index=False, name=None)
                )
                set_b = set(
                    modality_edges.loc[
                        (modality_edges["model_seed"] == seed_b)
                        & (modality_edges["rank"] <= top_k),
                        ["source", "target"],
                    ].itertuples(index=False, name=None)
                )
                intersection = len(set_a & set_b)
                union = len(set_a | set_b)
                smaller = min(len(set_a), len(set_b))
                rows.append(
                    {
                        "modality": modality,
                        "model_seed_a": int(seed_a),
                        "model_seed_b": int(seed_b),
                        "top_k": int(top_k),
                        "set_size_a": len(set_a),
                        "set_size_b": len(set_b),
                        "intersection_count": intersection,
                        "jaccard_index": intersection / union if union else 1.0,
                        "overlap_coefficient": intersection / smaller if smaller else 1.0,
                    }
                )
    pairwise = pd.DataFrame(rows)
    if pairwise.empty:
        return pairwise, pd.DataFrame()
    summary = pairwise.groupby(["modality", "top_k"], as_index=False).agg(
        pair_count=("jaccard_index", "size"),
        mean_intersection_count=("intersection_count", "mean"),
        std_intersection_count=("intersection_count", lambda values: values.std(ddof=0)),
        mean_jaccard_index=("jaccard_index", "mean"),
        std_jaccard_index=("jaccard_index", lambda values: values.std(ddof=0)),
        mean_overlap_coefficient=("overlap_coefficient", "mean"),
        std_overlap_coefficient=("overlap_coefficient", lambda values: values.std(ddof=0)),
    )
    return pairwise, summary


def compute_pairwise_rank_correlation(
    ranked_edges: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    for (modality, target), outcome_edges in ranked_edges.groupby(
        ["modality", "target"], sort=True
    ):
        pivot = outcome_edges.pivot_table(
            index="source", columns="model_seed", values="abs_edge_weight", aggfunc="first"
        )
        for seed_a, seed_b in itertools.combinations(sorted(pivot.columns), 2):
            paired = pivot[[seed_a, seed_b]].dropna()
            correlation = paired[seed_a].corr(paired[seed_b], method="spearman")
            rows.append(
                {
                    "modality": modality,
                    "target": target,
                    "model_seed_a": int(seed_a),
                    "model_seed_b": int(seed_b),
                    "edge_count": int(len(paired)),
                    "spearman_rank_correlation": float(correlation),
                }
            )
    pairwise = pd.DataFrame(rows)
    if pairwise.empty:
        return pairwise, pd.DataFrame()
    summary = pairwise.groupby(["modality", "target"], as_index=False).agg(
        pair_count=("spearman_rank_correlation", "size"),
        mean_spearman_rank_correlation=("spearman_rank_correlation", "mean"),
        std_spearman_rank_correlation=(
            "spearman_rank_correlation", lambda values: values.std(ddof=0)
        ),
    )
    return pairwise, summary


def build_consensus_edges(
    aggregate: pd.DataFrame,
    sign_threshold: float = 0.8,
    top100_frequency_threshold: float = 0.6,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    strict = aggregate.loc[
        (aggregate["sign_consistency"] >= sign_threshold)
        & (aggregate["top100_frequency"] >= top100_frequency_threshold)
    ].copy()
    strict = strict.sort_values(
        ["top100_frequency", "sign_consistency", "mean_abs_effect", "modality", "target", "source"],
        ascending=[False, False, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    ranked = aggregate.copy()
    rank_penalty = (
        ranked["mean_rank"].rank(method="average", pct=True)
        + ranked["median_rank"].rank(method="average", pct=True)
    ) / 2.0
    variability = ranked["effect_cv"].copy()
    fallback_variability = ranked["std_effect"] / ranked["mean_abs_effect"].clip(lower=1e-12)
    variability = variability.fillna(fallback_variability).replace([np.inf, -np.inf], np.nan)
    variability = variability.fillna(float(variability.max()) if variability.notna().any() else 1.0)
    variability_penalty = variability.rank(method="average", pct=True)
    ranked["rank_consensus_score"] = (
        0.50 * (1.0 - rank_penalty)
        + 0.30 * ranked["sign_consistency"]
        + 0.20 * (1.0 - variability_penalty)
    )
    ranked = ranked.sort_values(
        ["rank_consensus_score", "sign_consistency", "median_rank", "mean_rank", "modality", "target", "source"],
        ascending=[False, False, True, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "rank_consensus_order", np.arange(1, len(ranked) + 1))
    return strict, ranked


def load_outcome_decoder_norms(
    path: Path,
    outcomes: Iterable[str],
    model_seed: int,
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["model_seed", "target", "outcome_decoder_norm"])
    decoder = pd.read_csv(path, index_col=0)
    decoder = decoder.apply(pd.to_numeric, errors="coerce")
    rows = []
    for outcome in sorted(set(outcomes)):
        if outcome in decoder.index:
            rows.append(
                {
                    "model_seed": int(model_seed),
                    "target": outcome,
                    "outcome_decoder_norm": float(np.linalg.norm(decoder.loc[outcome].to_numpy(dtype=float))),
                }
            )
    return pd.DataFrame(rows)


def build_outcome_edge_distribution(
    ranked_edges: pd.DataFrame,
    strict_consensus: pd.DataFrame,
    decoder_norms: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    rows = []
    for target, group in ranked_edges.groupby("target", sort=True):
        total_strength = float(group["abs_edge_weight"].sum())
        protein = group.loc[group["modality"] == "protein"]
        metabolite = group.loc[group["modality"] == "metabolite"]
        top100_unique = group.loc[group["rank"] <= 100, ["modality", "source", "target"]].drop_duplicates()
        strict_count = int((strict_consensus["target"] == target).sum()) if not strict_consensus.empty else 0
        protein_strength = float(protein["abs_edge_weight"].sum())
        metabolite_strength = float(metabolite["abs_edge_weight"].sum())
        rows.append(
            {
                "target": target,
                "mean_absolute_edge_strength": float(group["abs_edge_weight"].mean()),
                "median_absolute_edge_strength": float(group["abs_edge_weight"].median()),
                "top100_edge_count": int(len(top100_unique)),
                "mean_top100_edge_count_per_seed": float((group["rank"] <= 100).sum() / group["model_seed"].nunique()),
                "consensus_edge_count": strict_count,
                "protein_contribution": protein_strength / total_strength if total_strength else 0.0,
                "metabolite_contribution": metabolite_strength / total_strength if total_strength else 0.0,
                "protein_mean_absolute_edge_strength": float(protein["abs_edge_weight"].mean()) if not protein.empty else np.nan,
                "metabolite_mean_absolute_edge_strength": float(metabolite["abs_edge_weight"].mean()) if not metabolite.empty else np.nan,
            }
        )
    outcome_summary = pd.DataFrame(rows)
    if decoder_norms is not None and not decoder_norms.empty:
        norm_summary = decoder_norms.groupby("target", as_index=False).agg(
            outcome_decoder_norm=("outcome_decoder_norm", "mean"),
            outcome_decoder_norm_std=("outcome_decoder_norm", lambda values: values.std(ddof=0)),
        )
        outcome_summary = outcome_summary.merge(norm_summary, on="target", how="left")
    else:
        outcome_summary["outcome_decoder_norm"] = np.nan
        outcome_summary["outcome_decoder_norm_std"] = np.nan
    outcome_summary["mean_absolute_edge_strength_per_decoder_norm"] = (
        outcome_summary["mean_absolute_edge_strength"]
        / outcome_summary["outcome_decoder_norm"].replace(0.0, np.nan)
    )
    return outcome_summary


def discover_runs(base_out_dir: Path) -> List[Tuple[int, Path]]:
    base_out_dir = Path(base_out_dir)
    runs = []
    if not base_out_dir.exists():
        return runs
    pattern = re.compile(r"^(?:model_seed|seed)_(\d+)$")
    for candidate in base_out_dir.iterdir():
        match = pattern.match(candidate.name) if candidate.is_dir() else None
        if match:
            runs.append((int(match.group(1)), candidate))
    return sorted(runs)


def _check_split_consistency(run_dirs: Sequence[Path]) -> Optional[bool]:
    split_keys = (
        "protein_train_eids", "protein_val_eids", "metabolite_train_eids",
        "metabolite_val_eids", "paired_train_eids", "paired_val_eids",
    )
    observed = []
    for run_dir in run_dirs:
        path = run_dir / "data_split_summary.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not all(key in metadata for key in split_keys):
            return None
        observed.append(tuple(tuple(metadata[key]) for key in split_keys))
    return len(set(observed)) <= 1


def analyze_experiment(
    base_out_dir: Path,
    output_dir: Path,
    experiment_name: str,
    fallback_split_seed: Optional[int] = None,
) -> Dict[str, object]:
    runs = discover_runs(base_out_dir)
    if not runs:
        raise FileNotFoundError(f"No seed result directories found in {base_out_dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_rows = []
    edge_frames = []
    decoder_frames = []
    successful_runs = []
    for fallback_model_seed, run_dir in runs:
        required = (
            run_dir / "run_summary.json",
            run_dir / "training_history.csv",
            run_dir / "G_protein_to_outcome.csv",
            run_dir / "G_metabolite_to_outcome.csv",
        )
        missing = [path.name for path in required if not path.exists()]
        if missing:
            print(f"Warning: skipping {run_dir}; missing {missing}")
            continue
        training = extract_training_summary(
            run_dir,
            fallback_model_seed=fallback_model_seed,
            fallback_split_seed=(
                fallback_split_seed if fallback_split_seed is not None else fallback_model_seed
            ),
        )
        model_seed = int(training["model_seed"])
        protein = load_edge_file(run_dir / "G_protein_to_outcome.csv", "protein", model_seed)
        metabolite = load_edge_file(run_dir / "G_metabolite_to_outcome.csv", "metabolite", model_seed)
        combined = pd.concat([protein, metabolite], ignore_index=True)
        training_rows.append(training)
        edge_frames.append(combined)
        decoder_frames.append(
            load_outcome_decoder_norms(
                run_dir / "decoder_projection_D.csv",
                outcomes=combined["target"].unique(),
                model_seed=model_seed,
            )
        )
        successful_runs.append(run_dir)
    if not edge_frames:
        raise RuntimeError(f"No complete runs found in {base_out_dir}")

    training_summary = flag_training_outliers(pd.DataFrame(training_rows).sort_values("model_seed"))
    ranked_edges = add_edge_ranks(pd.concat(edge_frames, ignore_index=True))
    model_seeds = sorted(training_summary["model_seed"].astype(int).unique())
    aggregate = aggregate_edge_statistics(ranked_edges, model_seeds=model_seeds)
    strict, rank_consensus = build_consensus_edges(aggregate)
    pairwise_topk, pairwise_topk_summary = compute_pairwise_topk_stability(ranked_edges)
    pairwise_rank, pairwise_rank_summary = compute_pairwise_rank_correlation(ranked_edges)
    decoder_norms = pd.concat(decoder_frames, ignore_index=True) if decoder_frames else pd.DataFrame()
    outcome_summary = build_outcome_edge_distribution(ranked_edges, strict, decoder_norms)

    aggregate = aggregate.sort_values(
        ["top100_frequency", "sign_consistency", "mean_abs_effect", "modality", "target", "source"],
        ascending=[False, False, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    training_summary.to_csv(output_dir / "multiseed_training_summary.csv", index=False)
    aggregate.to_csv(output_dir / "multiseed_edge_stability.csv", index=False)
    pairwise_topk.to_csv(output_dir / "pairwise_topk_stability.csv", index=False)
    pairwise_topk_summary.to_csv(output_dir / "pairwise_topk_stability_summary.csv", index=False)
    pairwise_rank.to_csv(output_dir / "pairwise_rank_correlation.csv", index=False)
    pairwise_rank_summary.to_csv(output_dir / "pairwise_rank_correlation_summary.csv", index=False)
    outcome_summary.to_csv(output_dir / "outcome_edge_distribution.csv", index=False)
    strict.to_csv(output_dir / "consensus_edges_strict.csv", index=False)
    rank_consensus.to_csv(output_dir / "consensus_edges_ranked.csv", index=False)
    strict.to_csv(output_dir / "consensus_outcome_edges.csv", index=False)

    rank_overall = pairwise_rank.groupby("modality", as_index=False).agg(
        pair_outcome_count=("spearman_rank_correlation", "size"),
        mean_spearman_rank_correlation=("spearman_rank_correlation", "mean"),
        std_spearman_rank_correlation=(
            "spearman_rank_correlation", lambda values: values.std(ddof=0)
        ),
    )
    rank_overall.to_csv(output_dir / "pairwise_rank_correlation_overall_summary.csv", index=False)

    def mean_topk(k: int) -> float:
        values = pairwise_topk.loc[pairwise_topk["top_k"] == k, "jaccard_index"]
        return float(values.mean()) if not values.empty else np.nan

    experiment_metrics = {
        "experiment": experiment_name,
        "run_count": len(model_seeds),
        "mean_top50_jaccard": mean_topk(50),
        "mean_top100_jaccard": mean_topk(100),
        "mean_spearman_correlation": float(pairwise_rank["spearman_rank_correlation"].mean()),
        "consensus_edge_count": int(len(strict)),
        "mean_sign_consistency": float(aggregate["sign_consistency"].mean()),
        "training_outlier_count": int(training_summary["is_training_outlier"].sum()),
        "patient_splits_identical": _check_split_consistency(successful_runs),
    }
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(experiment_metrics, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return experiment_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare variable-split and fixed-split multi-seed graph stability."
    )
    parser.add_argument(
        "--fixed-split-dir", default=str(DEFAULT_FIXED_SPLIT_DIR)
    )
    parser.add_argument(
        "--variable-split-dir", default=str(DEFAULT_VARIABLE_SPLIT_DIR)
    )
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_ANALYSIS_DIR)
    )
    parser.add_argument("--fixed-split-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments = (
        (
            "variable_split_experiment",
            Path(args.variable_split_dir),
            output_dir / "variable_split_experiment",
            None,
        ),
        (
            "fixed_split_experiment",
            Path(args.fixed_split_dir),
            output_dir / "fixed_split_experiment",
            args.fixed_split_seed,
        ),
    )
    comparison_rows = []
    for name, source, destination, fallback_split_seed in experiments:
        if not discover_runs(source):
            print(f"Warning: {name} is unavailable at {source}; skipping")
            continue
        metrics = analyze_experiment(
            source,
            destination,
            experiment_name=name,
            fallback_split_seed=fallback_split_seed,
        )
        comparison_rows.append(metrics)
        print(f"Saved {name} analysis to {destination}")
    if not comparison_rows:
        raise FileNotFoundError("Neither variable-split nor fixed-split results were found")
    pd.DataFrame(comparison_rows).to_csv(output_dir / "stability_comparison.csv", index=False)
    print(f"Saved comparison to {output_dir / 'stability_comparison.csv'}")


if __name__ == "__main__":
    main()
