"""Latent-SCM total propagation analysis for molecular-to-Outcome effects."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


EDGE_KEYS = ["modality", "outcome", "feature"]
HIGH_K = 100
TOP_KS = (20, 50, 100)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIMARY_DIR = PROJECT_ROOT / "artifacts/multiseed/fixed_split/model_seed_456"
DEFAULT_REPLICATION_DIRS = (
    PROJECT_ROOT / "artifacts/multiseed/variable_split/seed_456",
    PROJECT_ROOT / "artifacts/multiseed/variable_split/seed_789",
)
DEFAULT_SENSITIVITY_DIRS = (
    PROJECT_ROOT / "artifacts/multiseed/fixed_split/model_seed_789",
)
DEFAULT_DIRECT_COMPARISON = (
    PROJECT_ROOT
    / "artifacts/analysis/direct_effect_screening/outcome_specific_cross_model_comparison.csv"
)
DEFAULT_METABOLITE_DICTIONARY = (
    PROJECT_ROOT / "artifacts/metadata/metabolite_feature_dictionary.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/analysis/total_effect_propagation"
DEFAULT_VISUALIZATION_DIR = PROJECT_ROOT / "可视化/4.3"
OUTPUT_FILES = (
    "total_effect_primary.csv",
    "total_effect_cross_model_comparison.csv",
    "total_effect_top_edges.csv",
    "direct_vs_total_candidates.csv",
    "total_effect_stability_summary.csv",
    "multi_outcome_total_effect_summary.csv",
    "total_effect_summary.json",
)


def _validated_square_matrix(matrix: pd.DataFrame, source: str) -> pd.DataFrame:
    if matrix.empty or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{source} must be a non-empty square matrix")
    if matrix.index.has_duplicates or matrix.columns.has_duplicates:
        raise ValueError(f"{source} contains duplicate labels")
    if set(matrix.index.astype(str)) != set(matrix.columns.astype(str)):
        raise ValueError(f"{source} row and column labels differ")
    converted = matrix.apply(pd.to_numeric, errors="raise").copy()
    converted.index = converted.index.astype(str)
    converted.columns = converted.columns.astype(str)
    converted = converted.reindex(index=converted.columns)
    if not np.isfinite(converted.to_numpy(dtype=float)).all():
        raise ValueError(f"{source} contains NaN or infinite values")
    return converted


def compute_latent_total_propagation(
    adjacency: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Return (I-A)^-1-I and the requested stability diagnostics."""
    adjacency = _validated_square_matrix(adjacency, "latent adjacency A")
    values = adjacency.to_numpy(dtype=float)
    identity = np.eye(len(adjacency), dtype=float)
    system = identity - values
    eigenvalues = np.linalg.eigvals(values)
    spectral_radius = float(np.max(np.abs(eigenvalues)))
    condition_number = float(np.linalg.cond(system))
    if not np.isfinite(condition_number):
        raise ValueError("I-A has a non-finite condition number")
    try:
        propagation = np.linalg.solve(system, identity) - identity
    except np.linalg.LinAlgError as exc:
        raise ValueError("I-A is singular; total propagation cannot be computed") from exc
    if not np.isfinite(propagation).all():
        raise ValueError("latent total propagation contains NaN or infinite values")
    return (
        pd.DataFrame(
            propagation,
            index=adjacency.index.copy(),
            columns=adjacency.columns.copy(),
        ),
        {
            "spectral_radius_A": spectral_radius,
            "condition_number_I_minus_A": condition_number,
            "stable_spectral_radius": bool(spectral_radius < 1.0),
        },
    )


def _project_block(
    latent_matrix: pd.DataFrame,
    decoder: pd.DataFrame,
    outcomes: Sequence[str],
    features: Sequence[str],
) -> pd.DataFrame:
    values = (
        decoder.loc[list(outcomes), latent_matrix.index].to_numpy(dtype=float)
        @ latent_matrix.to_numpy(dtype=float)
        @ decoder.loc[list(features), latent_matrix.columns].to_numpy(dtype=float).T
    )
    return pd.DataFrame(values, index=list(outcomes), columns=list(features))


def project_molecular_outcome_blocks(
    adjacency: pd.DataFrame,
    total_latent: pd.DataFrame,
    decoder: pd.DataFrame,
    protein_features: Sequence[str],
    metabolite_features: Sequence[str],
    outcomes: Sequence[str],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Project direct and total effects without creating a full feature matrix."""
    adjacency = _validated_square_matrix(adjacency, "latent adjacency A")
    total_latent = _validated_square_matrix(total_latent, "latent total propagation")
    decoder = decoder.apply(pd.to_numeric, errors="raise").copy()
    decoder.index = decoder.index.astype(str)
    decoder.columns = decoder.columns.astype(str)
    if decoder.index.has_duplicates or decoder.columns.has_duplicates:
        raise ValueError("decoder projection D contains duplicate labels")
    if not np.isfinite(decoder.to_numpy(dtype=float)).all():
        raise ValueError("decoder projection D contains NaN or infinite values")
    if set(decoder.columns) != set(adjacency.index):
        raise ValueError("decoder latent columns do not match adjacency labels")
    decoder = decoder.reindex(columns=adjacency.index)
    total_latent = total_latent.reindex(index=adjacency.index, columns=adjacency.columns)
    requested = [*protein_features, *metabolite_features, *outcomes]
    missing = sorted(set(map(str, requested)).difference(decoder.index))
    if missing:
        raise ValueError(f"decoder projection is missing requested features: {missing[:20]}")

    blocks: Dict[str, Dict[str, pd.DataFrame]] = {}
    for modality, features in (
        ("Protein", protein_features),
        ("Metabolite", metabolite_features),
    ):
        blocks[modality] = {
            "direct": _project_block(adjacency, decoder, outcomes, features),
            "total": _project_block(total_latent, decoder, outcomes, features),
        }
    return blocks


def _rank_effect_column(edges: pd.DataFrame, effect_column: str) -> pd.Series:
    order = edges.assign(_abs=edges[effect_column].abs()).sort_values(
        ["modality", "outcome", "_abs", "feature"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    ranks = order.groupby(["modality", "outcome"], sort=False).cumcount() + 1
    return pd.Series(ranks.to_numpy(dtype=int), index=order.index).reindex(edges.index)


def build_model_edge_table(
    direct: pd.DataFrame,
    total: pd.DataFrame,
    modality: str,
    high_k: int = HIGH_K,
) -> pd.DataFrame:
    """Create signed direct/indirect/total effects and Outcome-specific ranks."""
    if high_k < 1:
        raise ValueError("high_k must be positive")
    if not direct.index.equals(total.index) or not direct.columns.equals(total.columns):
        raise ValueError("direct and total matrices must have identical labels")
    if not np.isfinite(direct.to_numpy(dtype=float)).all() or not np.isfinite(
        total.to_numpy(dtype=float)
    ).all():
        raise ValueError("direct or total matrix contains NaN or infinite values")
    direct_wide = direct.copy()
    direct_wide.index.name = "outcome"
    total_wide = total.copy()
    total_wide.index.name = "outcome"
    direct_edges = direct_wide.reset_index().melt(
        id_vars="outcome", var_name="feature", value_name="direct_effect"
    )
    total_edges = total_wide.reset_index().melt(
        id_vars="outcome", var_name="feature", value_name="total_effect"
    )
    edges = direct_edges.merge(
        total_edges, on=["outcome", "feature"], validate="one_to_one", sort=False
    )
    edges.insert(0, "modality", str(modality))
    edges["indirect_effect"] = edges["total_effect"] - edges["direct_effect"]
    edges["direct_outcome_rank"] = _rank_effect_column(edges, "direct_effect").astype(int)
    edges["total_outcome_rank"] = _rank_effect_column(edges, "total_effect").astype(int)
    edges["rank_shift"] = (
        edges["direct_outcome_rank"] - edges["total_outcome_rank"]
    ).astype(int)
    edges["direction_reversal"] = (
        edges["direct_effect"] * edges["total_effect"] < 0
    )
    direct_high = edges["direct_outcome_rank"] <= high_k
    total_high = edges["total_outcome_rank"] <= high_k
    edges["effect_pattern"] = np.select(
        [direct_high & total_high, ~direct_high & total_high, direct_high & ~total_high],
        ["direct_and_total_high", "indirect_amplified", "attenuated"],
        default="neither_high",
    )
    return edges.sort_values(
        ["modality", "outcome", "total_outcome_rank", "feature"], kind="mergesort"
    ).reset_index(drop=True)


def _validate_model_edges(
    model_edges: Mapping[str, pd.DataFrame], model_roles: Mapping[str, str]
) -> Tuple[str, ...]:
    if "primary" not in model_edges:
        raise ValueError("model_edges must contain primary")
    if set(model_edges) != set(model_roles):
        raise ValueError("model_edges and model_roles must have identical labels")
    other_labels = tuple(label for label in model_edges if label != "primary")
    if not other_labels:
        raise ValueError("at least one comparison model is required")
    required = {
        *EDGE_KEYS,
        "direct_effect",
        "indirect_effect",
        "total_effect",
        "direct_outcome_rank",
        "total_outcome_rank",
        "rank_shift",
        "direction_reversal",
        "effect_pattern",
    }
    for label, edges in model_edges.items():
        missing = sorted(required.difference(edges.columns))
        if missing:
            raise ValueError(f"{label} edge table is missing columns: {missing}")
        if edges.duplicated(EDGE_KEYS).any():
            raise ValueError(f"{label} contains duplicate feature-to-Outcome edges")
        numeric = edges[
            [
                "direct_effect",
                "indirect_effect",
                "total_effect",
                "direct_outcome_rank",
                "total_outcome_rank",
                "rank_shift",
            ]
        ].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"{label} edge table contains non-finite values")
    return other_labels


def build_cross_model_comparison(
    model_edges: Mapping[str, pd.DataFrame],
    model_roles: Mapping[str, str],
    top_ks: Iterable[int] = TOP_KS,
    high_k: int = HIGH_K,
) -> pd.DataFrame:
    """Join identical edges across models and calculate replication support."""
    top_ks = tuple(sorted({int(k) for k in top_ks}))
    if not top_ks or min(top_ks) < 1 or high_k < 1:
        raise ValueError("top_ks and high_k must be positive")
    other_labels = _validate_model_edges(model_edges, model_roles)
    labels = ("primary", *other_labels)
    fields = (
        "direct_effect",
        "indirect_effect",
        "total_effect",
        "direct_outcome_rank",
        "total_outcome_rank",
        "rank_shift",
    )
    comparison: pd.DataFrame | None = None
    for label in labels:
        selected = model_edges[label].loc[:, [*EDGE_KEYS, *fields]].rename(
            columns={field: f"{label}_{field}" for field in fields}
        )
        comparison = (
            selected
            if comparison is None
            else comparison.merge(
                selected,
                on=EDGE_KEYS,
                how="left",
                validate="one_to_one",
                sort=False,
            )
        )
    assert comparison is not None
    model_columns = [f"{label}_{field}" for label in labels for field in fields]
    if comparison[model_columns].isna().any().any():
        raise ValueError("one or more models are missing Primary edges")

    primary_sign = np.sign(comparison["primary_total_effect"].to_numpy(dtype=float))
    sign_columns = []
    for label in other_labels:
        column = f"{label}_total_sign_agrees_primary"
        comparison[column] = (
            np.sign(comparison[f"{label}_total_effect"].to_numpy(dtype=float))
            == primary_sign
        )
        sign_columns.append(column)
    comparison["total_sign_agreement_count"] = (
        comparison[sign_columns].sum(axis=1).astype(int)
    )
    comparison["total_sign_agreement_fraction"] = (
        comparison["total_sign_agreement_count"] / len(other_labels)
    )

    total_rank_columns = [f"{label}_total_outcome_rank" for label in labels]
    direct_rank_columns = [f"{label}_direct_outcome_rank" for label in labels]
    rank_shift_columns = [f"{label}_rank_shift" for label in labels]
    for k in top_ks:
        comparison[f"total_top{k}_support_count"] = (
            comparison[total_rank_columns].le(k).sum(axis=1).astype(int)
        )
    comparison["rank_shift_positive_support_count"] = (
        comparison[rank_shift_columns].gt(0).sum(axis=1).astype(int)
    )
    amplified = pd.concat(
        [
            comparison[direct_column].gt(high_k)
            & comparison[total_column].le(high_k)
            for direct_column, total_column in zip(
                direct_rank_columns, total_rank_columns
            )
        ],
        axis=1,
    )
    comparison["indirect_amplified_support_count"] = amplified.sum(axis=1).astype(int)
    return comparison.sort_values(
        ["modality", "outcome", "primary_total_outcome_rank", "feature"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_total_stability(
    model_edges: Mapping[str, pd.DataFrame],
    model_roles: Mapping[str, str],
    top_ks: Iterable[int] = TOP_KS,
) -> pd.DataFrame:
    """Measure Primary-vs-other Total rank stability by modality and Outcome."""
    top_ks = tuple(sorted({int(k) for k in top_ks}))
    if not top_ks or min(top_ks) < 1:
        raise ValueError("top_ks must contain positive values")
    other_labels = _validate_model_edges(model_edges, model_roles)
    rows = []
    primary = model_edges["primary"]
    for (modality, outcome), primary_group in primary.groupby(
        ["modality", "outcome"], sort=True
    ):
        primary_group = primary_group.sort_values("feature", kind="mergesort")
        for label in other_labels:
            other_group = model_edges[label].loc[
                (model_edges[label]["modality"] == modality)
                & (model_edges[label]["outcome"] == outcome),
                ["feature", "total_outcome_rank"],
            ]
            paired = primary_group.loc[:, ["feature", "total_outcome_rank"]].merge(
                other_group,
                on="feature",
                how="left",
                suffixes=("_primary", "_other"),
                validate="one_to_one",
                sort=False,
            )
            if len(paired) != len(primary_group) or paired.isna().any().any():
                raise ValueError(f"{label} {modality} {outcome} features differ from Primary")
            correlation = float(
                spearmanr(
                    paired["total_outcome_rank_primary"].to_numpy(dtype=float),
                    paired["total_outcome_rank_other"].to_numpy(dtype=float),
                ).statistic
            )
            if not np.isfinite(correlation):
                raise ValueError(f"undefined Total rank Spearman for {label} {modality} {outcome}")
            row = {
                "modality": modality,
                "outcome": outcome,
                "comparison_model": label,
                "comparison_role": model_roles[label],
                "feature_count": int(len(paired)),
                "total_rank_spearman": correlation,
            }
            for k in top_ks:
                primary_top = set(
                    paired.loc[paired["total_outcome_rank_primary"] <= k, "feature"]
                )
                other_top = set(
                    paired.loc[paired["total_outcome_rank_other"] <= k, "feature"]
                )
                intersection = len(primary_top & other_top)
                union = len(primary_top | other_top)
                row[f"total_top{k}_intersection"] = int(intersection)
                row[f"total_top{k}_jaccard"] = float(
                    intersection / union if union else 1.0
                )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["modality", "outcome", "comparison_model"], kind="mergesort"
    ).reset_index(drop=True)


def build_multi_outcome_summary(
    primary_edges: pd.DataFrame, top_ks: Iterable[int] = TOP_KS
) -> pd.DataFrame:
    """Summarize recurrent Total Top-K and propagation patterns per feature."""
    top_ks = tuple(sorted({int(k) for k in top_ks}))
    if not top_ks or min(top_ks) < 1:
        raise ValueError("top_ks must contain positive values")
    rows = []
    for (modality, feature), group in primary_edges.groupby(
        ["modality", "feature"], sort=True
    ):
        row = {"modality": modality, "feature": feature}
        for k in top_ks:
            outcomes = sorted(
                group.loc[group["total_outcome_rank"] <= k, "outcome"]
                .astype(str)
                .tolist()
            )
            row[f"n_outcomes_total_top{k}"] = int(len(outcomes))
            row[f"outcomes_total_top{k}"] = ";".join(outcomes) if outcomes else "none"
        amplified = sorted(
            group.loc[group["effect_pattern"] == "indirect_amplified", "outcome"]
            .astype(str)
            .tolist()
        )
        reversals = sorted(
            group.loc[group["direction_reversal"], "outcome"].astype(str).tolist()
        )
        row["n_outcomes_indirect_amplified"] = int(len(amplified))
        row["outcomes_indirect_amplified"] = ";".join(amplified) if amplified else "none"
        row["n_outcomes_direction_reversal"] = int(len(reversals))
        row["outcomes_direction_reversal"] = ";".join(reversals) if reversals else "none"
        rows.append(row)
    count_columns = [f"n_outcomes_total_top{k}" for k in top_ks]
    return pd.DataFrame(rows).sort_values(
        ["modality", *count_columns, "feature"],
        ascending=[True, *([False] * len(count_columns)), True],
        kind="mergesort",
    ).reset_index(drop=True)


def _load_labeled_matrix(path: Path | str) -> pd.DataFrame:
    matrix_path = Path(path)
    if not matrix_path.is_file():
        raise FileNotFoundError(f"Required matrix not found: {matrix_path}")
    with matrix_path.open("r", encoding="utf-8-sig", newline="") as handle:
        try:
            header = next(csv.reader(handle))
        except StopIteration as exc:
            raise ValueError(f"{matrix_path} is empty") from exc
    labels = [value.strip() for value in header[1:]]
    if not labels or any(not value for value in labels) or len(labels) != len(set(labels)):
        raise ValueError(f"{matrix_path} has blank or duplicate column labels")
    matrix = pd.read_csv(matrix_path, index_col=0)
    if matrix.empty or matrix.index.has_duplicates:
        raise ValueError(f"{matrix_path} is empty or has duplicate row labels")
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    matrix = matrix.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError(f"{matrix_path} contains NaN or infinite values")
    return matrix


def _model_specs(
    primary_dir: Path,
    replication_dirs: Sequence[Path],
    sensitivity_dirs: Sequence[Path],
) -> List[Dict[str, object]]:
    if len(replication_dirs) != 2:
        raise ValueError("4.3 requires exactly two replication directories")
    if len(sensitivity_dirs) != 1:
        raise ValueError("4.3 requires exactly one sensitivity directory")
    paths = [primary_dir, *replication_dirs, *sensitivity_dirs]
    resolved = [str(Path(path).resolve(strict=False)).casefold() for path in paths]
    if len(resolved) != len(set(resolved)):
        raise ValueError("all four model directories must be distinct")
    return [
        {"label": "primary", "role": "primary", "directory": Path(primary_dir)},
        {
            "label": "variable456",
            "role": "replication",
            "directory": Path(replication_dirs[0]),
        },
        {
            "label": "variable789",
            "role": "replication",
            "directory": Path(replication_dirs[1]),
        },
        {
            "label": "fixed789",
            "role": "sensitivity",
            "directory": Path(sensitivity_dirs[0]),
        },
    ]


def _assert_same_labels(
    actual: pd.DataFrame, expected: pd.DataFrame, label: str
) -> pd.DataFrame:
    if set(actual.index) != set(expected.index):
        raise ValueError(f"{label} Outcome labels differ from Primary")
    if set(actual.columns) != set(expected.columns):
        raise ValueError(f"{label} feature labels differ from Primary")
    return actual.reindex(index=expected.index, columns=expected.columns)


def _load_model_edges(
    specs: Sequence[Mapping[str, object]], high_k: int
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str], Dict[str, Dict[str, object]]]:
    model_edges: Dict[str, pd.DataFrame] = {}
    roles: Dict[str, str] = {}
    diagnostics: Dict[str, Dict[str, object]] = {}
    primary_blocks: Dict[str, pd.DataFrame] = {}
    for spec in specs:
        label = str(spec["label"])
        directory = Path(spec["directory"])
        adjacency = _load_labeled_matrix(directory / "latent_causal_A.csv")
        decoder = _load_labeled_matrix(directory / "decoder_projection_D.csv")
        adjacency = _validated_square_matrix(adjacency, f"{label} latent adjacency A")
        protein_direct = _load_labeled_matrix(directory / "G_protein_to_outcome.csv")
        metabolite_direct = _load_labeled_matrix(
            directory / "G_metabolite_to_outcome.csv"
        )
        if label == "primary":
            primary_blocks = {
                "Protein": protein_direct,
                "Metabolite": metabolite_direct,
            }
        else:
            protein_direct = _assert_same_labels(
                protein_direct, primary_blocks["Protein"], f"{label} Protein"
            )
            metabolite_direct = _assert_same_labels(
                metabolite_direct, primary_blocks["Metabolite"], f"{label} Metabolite"
            )
        if not protein_direct.index.equals(metabolite_direct.index):
            raise ValueError(f"{label} Protein and Metabolite Outcomes differ")

        total_latent, model_diagnostics = compute_latent_total_propagation(adjacency)
        projected = project_molecular_outcome_blocks(
            adjacency,
            total_latent,
            decoder,
            protein_direct.columns.tolist(),
            metabolite_direct.columns.tolist(),
            protein_direct.index.tolist(),
        )
        model_tables = []
        direct_errors = {}
        for modality, exported_direct in (
            ("Protein", protein_direct),
            ("Metabolite", metabolite_direct),
        ):
            computed_direct = projected[modality]["direct"]
            error = float(
                np.max(
                    np.abs(
                        computed_direct.to_numpy(dtype=float)
                        - exported_direct.to_numpy(dtype=float)
                    )
                )
            )
            direct_errors[modality] = error
            if not np.allclose(
                computed_direct.to_numpy(dtype=float),
                exported_direct.to_numpy(dtype=float),
                rtol=1e-5,
                atol=1e-8,
            ):
                raise ValueError(
                    f"{label} {modality} DAD^T does not match exported direct G; "
                    f"max_abs_error={error:.6g}"
                )
            model_tables.append(
                build_model_edge_table(
                    exported_direct, projected[modality]["total"], modality, high_k
                )
            )
        edges = pd.concat(model_tables, ignore_index=True)
        expected_count = len(protein_direct.index) * (
            len(protein_direct.columns) + len(metabolite_direct.columns)
        )
        if len(edges) != expected_count:
            raise ValueError(
                f"{label} produced {len(edges)} edges instead of {expected_count}"
            )
        model_edges[label] = edges
        roles[label] = str(spec["role"])
        diagnostics[label] = {
            **model_diagnostics,
            "direct_projection_max_abs_error": direct_errors,
            "edge_count": int(len(edges)),
        }
    return model_edges, roles, diagnostics


def _validate_against_direct_comparison(
    model_edges: Mapping[str, pd.DataFrame], direct_path: Path | str
) -> Dict[str, Dict[str, float]]:
    path = Path(direct_path)
    if not path.is_file():
        raise FileNotFoundError(f"4.2 direct comparison not found: {path}")
    reference = pd.read_csv(path)
    missing_keys = sorted(set(EDGE_KEYS).difference(reference.columns))
    if missing_keys:
        raise ValueError(f"4.2 direct comparison is missing keys: {missing_keys}")
    if reference.duplicated(EDGE_KEYS).any():
        raise ValueError("4.2 direct comparison contains duplicate edges")
    reference_keys = set(
        map(tuple, reference.loc[:, EDGE_KEYS].astype(str).to_numpy())
    )
    generated_keys = set(
        map(
            tuple,
            model_edges["primary"].loc[:, EDGE_KEYS].astype(str).to_numpy(),
        )
    )
    if reference_keys != generated_keys:
        missing = len(generated_keys - reference_keys)
        extra = len(reference_keys - generated_keys)
        raise ValueError(
            f"4.2 and 4.3 edge keys differ: missing={missing}, extra={extra}"
        )
    report: Dict[str, Dict[str, float]] = {}
    for label, edges in model_edges.items():
        effect_column = f"{label}_effect"
        rank_column = f"{label}_outcome_rank"
        missing = [column for column in (effect_column, rank_column) if column not in reference]
        if missing:
            raise ValueError(f"4.2 direct comparison is missing columns: {missing}")
        reference_ranks = pd.to_numeric(reference[rank_column], errors="raise")
        if (
            not np.isfinite(reference_ranks.to_numpy(dtype=float)).all()
            or (reference_ranks < 1).any()
            or not np.equal(reference_ranks, np.floor(reference_ranks)).all()
        ):
            raise ValueError(
                f"4.2 direct comparison {rank_column} must contain positive integer ranks"
            )
        paired = edges.loc[:, [*EDGE_KEYS, "direct_effect", "direct_outcome_rank"]].merge(
            reference.loc[:, [*EDGE_KEYS, effect_column, rank_column]],
            on=EDGE_KEYS,
            how="left",
            validate="one_to_one",
            sort=False,
        )
        if len(paired) != len(edges) or paired[[effect_column, rank_column]].isna().any().any():
            raise ValueError(f"4.2 direct comparison does not contain every {label} edge")
        effect_error = float(
            np.max(
                np.abs(
                    paired["direct_effect"].to_numpy(dtype=float)
                    - paired[effect_column].to_numpy(dtype=float)
                )
            )
        )
        rank_mismatch = int(
            np.sum(
                paired["direct_outcome_rank"].to_numpy(dtype=int)
                != paired[rank_column].to_numpy(dtype=int)
            )
        )
        if not np.allclose(
            paired["direct_effect"].to_numpy(dtype=float),
            paired[effect_column].to_numpy(dtype=float),
            rtol=1e-7,
            atol=1e-10,
        ) or rank_mismatch:
            raise ValueError(
                f"{label} direct effects/ranks disagree with 4.2: "
                f"max_abs_error={effect_error:.6g}, rank_mismatches={rank_mismatch}"
            )
        report[label] = {
            "max_abs_effect_error_vs_4_2": effect_error,
            "rank_mismatch_count_vs_4_2": rank_mismatch,
        }
    return report


def _load_metabolite_dictionary(path: Path | str) -> pd.DataFrame:
    dictionary_path = Path(path)
    if not dictionary_path.is_file():
        raise FileNotFoundError(f"Metabolite dictionary not found: {dictionary_path}")
    dictionary = pd.read_csv(dictionary_path, dtype=str, keep_default_na=False)
    required = ["feature_code", "metabolite_name"]
    missing = [column for column in required if column not in dictionary.columns]
    if missing:
        raise ValueError(f"Metabolite dictionary is missing columns: {missing}")
    dictionary = dictionary.loc[:, required]
    if dictionary["feature_code"].duplicated().any() or dictionary.eq("").any().any():
        raise ValueError("Metabolite dictionary contains duplicate or blank values")
    return dictionary


def _annotate_metabolites(
    frame: pd.DataFrame, dictionary: pd.DataFrame
) -> pd.DataFrame:
    annotated = frame.copy()
    annotated.insert(3, "metabolite_name", "not_applicable")
    mask = annotated["modality"].eq("Metabolite")
    mapping = dictionary.set_index("feature_code")["metabolite_name"]
    names = annotated.loc[mask, "feature"].astype(str).map(mapping)
    if names.isna().any():
        missing = sorted(annotated.loc[mask, "feature"][names.isna()].unique())
        raise ValueError(f"Metabolite dictionary has no mapping for: {missing[:20]}")
    annotated.loc[mask, "metabolite_name"] = names.to_numpy()
    return annotated


def _validate_output_frame(frame: pd.DataFrame, label: str) -> None:
    if frame.empty:
        raise ValueError(f"{label} is empty")
    if frame.isna().any().any():
        raise ValueError(f"{label} contains missing values")
    numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{label} contains NaN or infinite numeric values")


def _pattern_summary(primary_edges: pd.DataFrame) -> Dict[str, object]:
    overall = {
        pattern: int((primary_edges["effect_pattern"] == pattern).sum())
        for pattern in (
            "direct_and_total_high",
            "indirect_amplified",
            "attenuated",
            "neither_high",
        )
    }
    overall["direction_reversal"] = int(primary_edges["direction_reversal"].sum())
    by_outcome = []
    for (modality, outcome), group in primary_edges.groupby(
        ["modality", "outcome"], sort=True
    ):
        by_outcome.append(
            {
                "modality": modality,
                "outcome": outcome,
                "direct_and_total_high": int(
                    (group["effect_pattern"] == "direct_and_total_high").sum()
                ),
                "indirect_amplified": int(
                    (group["effect_pattern"] == "indirect_amplified").sum()
                ),
                "attenuated": int((group["effect_pattern"] == "attenuated").sum()),
                "neither_high": int(
                    (group["effect_pattern"] == "neither_high").sum()
                ),
                "direction_reversal": int(group["direction_reversal"].sum()),
            }
        )
    return {"overall": overall, "by_modality_outcome": by_outcome}


def _summary_payload(
    specs: Sequence[Mapping[str, object]],
    diagnostics: Mapping[str, Mapping[str, object]],
    direct_validation: Mapping[str, Mapping[str, float]],
    primary_edges: pd.DataFrame,
    comparison: pd.DataFrame,
    stability: pd.DataFrame,
    args: argparse.Namespace,
) -> Dict[str, object]:
    models: Dict[str, object] = {"replication": [], "sensitivity": []}
    for spec in specs:
        label = str(spec["label"])
        item = {
            "label": label,
            "role": str(spec["role"]),
            "directory": str(Path(spec["directory"]).resolve()),
            **diagnostics[label],
            **direct_validation[label],
        }
        if spec["role"] == "primary":
            models["primary"] = item
        else:
            models[str(spec["role"])].append(item)
    feature_counts = {
        modality: int(
            primary_edges.loc[primary_edges["modality"] == modality, "feature"].nunique()
        )
        for modality in ("Protein", "Metabolite")
    }
    stability_means = (
        stability.groupby(["modality", "outcome"], sort=True)[
            ["total_rank_spearman", *[f"total_top{k}_jaccard" for k in TOP_KS]]
        ]
        .mean()
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "analysis": "4.3 Total Structural Propagation Effect Analysis",
        "analysis_unit": "feature_to_outcome_edge",
        "latent_total_definition": "T_A=(I-A)^-1-I",
        "feature_total_definition": "G_T=D*T_A*D^T; only molecular-to-Outcome blocks retained",
        "models": models,
        "outcomes": sorted(primary_edges["outcome"].unique().tolist()),
        "outcome_count": int(primary_edges["outcome"].nunique()),
        "feature_counts": feature_counts,
        "primary_edge_count": int(len(primary_edges)),
        "cross_model_edge_count": int(len(comparison)),
        "top_k_thresholds": list(TOP_KS),
        "high_rank_threshold": int(args.high_k),
        "top_edges_per_outcome": int(args.top_edges_per_outcome),
        "ranking_rule": "within modality and Outcome: abs(effect) descending, feature ascending for ties",
        "total_sign_agreement_denominator": 3,
        "top_k_support_denominator": 4,
        "effect_pattern_counts": _pattern_summary(primary_edges),
        "outcome_stability_mean_across_other_models": stability_means,
        "direct_comparison_4_2": str(Path(args.direct_comparison).resolve()),
        "metabolite_dictionary": str(Path(args.metabolite_dictionary).resolve()),
        "output_files": list(OUTPUT_FILES),
    }


DISPLAY_COLUMN_LABELS = {
    "modality": "模态",
    "feature_display": "特征",
    "direct_effect": "直接效应",
    "total_effect": "总效应",
    "direct_outcome_rank": "直接效应排名",
    "total_outcome_rank": "总效应排名",
    "rank_shift": "排名变化",
    "total_sign_agreement_count": "总效应同号支持（/3）",
    "total_top20_support_count": "Total Top20 支持（/4）",
    "indirect_amplified_support_count": "间接放大支持（/4）",
    "n_outcomes_total_top20": "进入 Total Top20 的 Outcome 数",
    "outcomes_total_top20": "Total Top20 Outcomes",
    "n_outcomes_total_top50": "进入 Total Top50 的 Outcome 数",
    "outcomes_total_top50": "Total Top50 Outcomes",
    "n_outcomes_indirect_amplified": "间接放大 Outcome 数",
    "n_outcomes_direction_reversal": "方向翻转 Outcome 数",
}


def _with_visual_feature_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Use readable metabolite names without exposing a separate name column."""
    visual = frame.copy()
    visual["feature_display"] = visual["feature"].astype(str)
    metabolite_mask = visual["modality"].eq("Metabolite")
    visual.loc[metabolite_mask, "feature_display"] = (
        visual.loc[metabolite_mask, "metabolite_name"].astype(str)
        + " ["
        + visual.loc[metabolite_mask, "feature"].astype(str)
        + "]"
    )
    return visual


def _table_html(frame: pd.DataFrame, columns: Sequence[str], limit: int = 100) -> str:
    selected = frame.loc[:, [column for column in columns if column in frame.columns]].head(limit)
    selected = selected.rename(columns=DISPLAY_COLUMN_LABELS)
    return selected.to_html(index=False, border=0, classes="data-table", escape=True)


def _page_shell(title: str, content: str, scripts: str = "") -> str:
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root{{--bg:#f7f8fa;--surface:#fff;--text:#172033;--muted:#667085;--line:#d9dee8;--accent:#315efb}}
@media(prefers-color-scheme:dark){{:root{{--bg:#10131a;--surface:#171b24;--text:#edf1f7;--muted:#a5adba;--line:#343b49;--accent:#83a0ff}}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}}
main{{max-width:1500px;margin:auto;padding:24px}} h1,h2{{font-weight:600}} a{{color:var(--accent)}}
.nav{{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0 22px}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:22px}}
.panel{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px;overflow:auto}}
.data-table{{border-collapse:collapse;width:100%;font-size:12px}} .data-table th,.data-table td{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}
.data-table th{{position:sticky;top:0;background:var(--surface)}} .plot{{min-height:430px}} .muted{{color:var(--muted)}}
@media(max-width:600px){{main{{padding:12px}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>{content}</main>{scripts}</body></html>"""


def _json_for_script(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def _outcome_filename(outcome: str) -> str:
    original = str(outcome)
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", original).strip("_") or "unnamed"
    if safe != original:
        digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:8]
        safe = f"{safe}_{digest}"
    return f"outcome_{safe}.html"


def render_visualizations(
    primary: pd.DataFrame,
    comparison: pd.DataFrame,
    stability: pd.DataFrame,
    multi: pd.DataFrame,
    output_dir: Path | str,
) -> None:
    """Write a lightweight standalone index and one page per Outcome."""
    visual_dir = Path(output_dir)
    visual_dir.mkdir(parents=True, exist_ok=True)
    outcomes = sorted(primary["outcome"].unique().tolist())
    links = "".join(
        f'<a href="{html.escape(_outcome_filename(outcome))}">{html.escape(outcome)}</a>'
        for outcome in outcomes
    )
    stability_view = stability.copy()
    stability_view["row_label"] = (
        stability_view["modality"].astype(str)
        + " · "
        + stability_view["outcome"].astype(str)
    )
    comparisons = sorted(stability_view["comparison_model"].unique().tolist())
    rows = sorted(stability_view["row_label"].unique().tolist())
    heatmap = (
        stability_view.pivot(
            index="row_label", columns="comparison_model", values="total_rank_spearman"
        )
        .reindex(index=rows, columns=comparisons)
        .to_numpy(dtype=float)
        .tolist()
    )
    multi_view = _with_visual_feature_labels(multi.loc[
        (multi["n_outcomes_total_top20"] > 0)
        | (multi["n_outcomes_total_top50"] > 0)
    ].sort_values(
        ["n_outcomes_total_top20", "n_outcomes_total_top50", "modality", "feature"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ))
    index_content = f"""
<h1>4.3 总结构传播效应分析</h1>
<p class="muted">以 Primary 模型为正式结果，并与两个 Replication 模型和一个 Sensitivity 模型比较。请选择 Outcome 查看总效应 Top20 及 Direct→Total 变化。</p>
<nav class="nav">{links}</nav>
<section class="panel"><h2>总传播效应排名的跨模型稳定性</h2><p class="muted">用于观察同一 Outcome 的总效应排序在不同模型间是否一致。热图显示 Primary 与各比较模型的 Spearman 排名相关；数值越接近 1，整体排序越稳定。</p><div id="stability-heatmap" class="plot"></div></section>
<section class="panel"><h2>多 Outcome 总效应候选</h2><p class="muted">统计同一特征在多个 Outcome 中重复进入 Total Top-K 的情况，同时给出间接放大和方向翻转出现次数。该栏目用于发现跨 Outcome 重复信号，不单独定义 biomarker。</p>{_table_html(multi_view, ['modality','feature_display','n_outcomes_total_top20','outcomes_total_top20','n_outcomes_total_top50','outcomes_total_top50','n_outcomes_indirect_amplified','n_outcomes_direction_reversal'], 200)}</section>
"""
    index_scripts = f"""
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<script>Plotly.newPlot('stability-heatmap',[{{type:'heatmap',z:{_json_for_script(heatmap)},x:{_json_for_script(comparisons)},y:{_json_for_script(rows)},zmin:-1,zmax:1,colorscale:'RdBu',reversescale:true,colorbar:{{title:'Spearman'}}}}],{{margin:{{l:180,r:30,t:20,b:70}},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{{color:getComputedStyle(document.documentElement).getPropertyValue('--text')}}}},{{responsive:true}});</script>
"""
    (visual_dir / "index.html").write_text(
        _page_shell("4.3 总结构传播效应分析", index_content, index_scripts),
        encoding="utf-8",
    )

    table_columns = [
        "feature_display",
        "direct_effect",
        "total_effect",
        "direct_outcome_rank",
        "total_outcome_rank",
        "rank_shift",
        "total_sign_agreement_count",
        "total_top20_support_count",
        "indirect_amplified_support_count",
    ]
    for outcome in outcomes:
        outcome_primary = primary.loc[primary["outcome"] == outcome]
        outcome_comparison = comparison.loc[comparison["outcome"] == outcome]
        top_sections = []
        scatter_traces = []
        for modality in ("Protein", "Metabolite"):
            modality_title = "蛋白质" if modality == "Protein" else "代谢物"
            top = _with_visual_feature_labels(outcome_comparison.loc[
                (outcome_comparison["modality"] == modality)
                & (outcome_comparison["primary_total_outcome_rank"] <= 20)
            ].rename(
                columns={
                    "primary_direct_effect": "direct_effect",
                    "primary_total_effect": "total_effect",
                    "primary_direct_outcome_rank": "direct_outcome_rank",
                    "primary_total_outcome_rank": "total_outcome_rank",
                    "primary_rank_shift": "rank_shift",
                }
            ))
            top_description = (
                "按总效应绝对值从高到低列出该 Outcome 的前 20 个蛋白质。"
                if modality == "Protein"
                else "按总效应绝对值从高到低列出该 Outcome 的前 20 个代谢物；特征列直接显示真实名称，并在方括号中保留原始编码。"
            )
            top_sections.append(
                f'<section class="panel"><h2>{modality_title}总效应 Top20</h2><p class="muted">{top_description}</p>{_table_html(top, table_columns, 20)}</section>'
            )
            points = _with_visual_feature_labels(
                outcome_primary.loc[outcome_primary["modality"] == modality]
            )
            scatter_traces.append(
                {
                    "type": "scattergl",
                    "mode": "markers",
                    "name": modality,
                    "x": points["direct_outcome_rank"].astype(int).tolist(),
                    "y": points["total_outcome_rank"].astype(int).tolist(),
                    "text": points["feature_display"].astype(str).tolist(),
                    "customdata": points["rank_shift"].astype(int).tolist(),
                    "hovertemplate": "%{text}<br>直接效应排名=%{x}<br>总效应排名=%{y}<br>排名变化=%{customdata}<extra></extra>",
                }
            )
        candidate_sections = []
        for pattern, heading, description in (
            (
                "indirect_amplified",
                "间接放大候选",
                "直接效应排名大于 100、总效应排名进入前 100，表示网络传播后该特征的重要性上升。跨模型支持较低时仍应视为探索性线索。",
            ),
            (
                "attenuated",
                "衰减候选",
                "直接效应排名在前 100、总效应排名跌出前 100，表示间接传播可能削弱或抵消直接结构效应。",
            ),
        ):
            candidates = _with_visual_feature_labels(outcome_comparison.loc[
                outcome_comparison["effect_pattern"] == pattern
            ].rename(
                columns={
                    "primary_direct_effect": "direct_effect",
                    "primary_total_effect": "total_effect",
                    "primary_direct_outcome_rank": "direct_outcome_rank",
                    "primary_total_outcome_rank": "total_outcome_rank",
                    "primary_rank_shift": "rank_shift",
                }
            ))
            candidate_sections.append(
                f'<section class="panel"><h2>{heading}</h2><p class="muted">{description}</p>{_table_html(candidates, ["modality", *table_columns], 100)}</section>'
            )
        reversals = _with_visual_feature_labels(outcome_comparison.loc[outcome_comparison["direction_reversal"]].rename(
            columns={
                "primary_direct_effect": "direct_effect",
                "primary_total_effect": "total_effect",
                "primary_direct_outcome_rank": "direct_outcome_rank",
                "primary_total_outcome_rank": "total_outcome_rank",
                "primary_rank_shift": "rank_shift",
            }
        ))
        content = f"""
<a href="index.html">← 返回全部 Outcome</a><h1>{html.escape(str(outcome))}：直接效应与总效应对比</h1>
<div class="grid">{''.join(top_sections)}</div>
<section class="panel"><h2>直接效应排名与总效应排名</h2><p class="muted">横轴为直接效应排名，纵轴为总效应排名，均使用对数刻度。将同一特征的两个排名对照，可判断网络传播后排名上升、下降或基本不变；悬浮可查看具体特征和 RankShift。</p><div id="rank-scatter" class="plot"></div></section>
<div class="grid">{''.join(candidate_sections)}</div>
<section class="panel"><h2>方向翻转候选</h2><p class="muted">直接效应与总效应符号相反，表示完整网络传播后效应方向发生翻转。应结合效应量和排名判断，尾部弱效应翻转不宜过度解释。</p>{_table_html(reversals, ["modality", *table_columns], 150)}</section>
"""
        scripts = f"""
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<script>Plotly.newPlot('rank-scatter',{_json_for_script(scatter_traces)},{{xaxis:{{title:'直接效应 Outcome 排名',type:'log',autorange:'reversed'}},yaxis:{{title:'总效应 Outcome 排名',type:'log',autorange:'reversed'}},margin:{{l:70,r:30,t:20,b:65}},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{{color:getComputedStyle(document.documentElement).getPropertyValue('--text')}}}},{{responsive:true}});</script>
"""
        (visual_dir / _outcome_filename(outcome)).write_text(
            _page_shell(f"4.3 {outcome}", content, scripts), encoding="utf-8"
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run 4.3 latent-SCM total structural propagation analysis without retraining."
    )
    parser.add_argument("--primary_dir", type=Path, default=DEFAULT_PRIMARY_DIR)
    parser.add_argument(
        "--replication_dirs", type=Path, nargs=2, default=DEFAULT_REPLICATION_DIRS
    )
    parser.add_argument(
        "--sensitivity_dirs", type=Path, nargs=1, default=DEFAULT_SENSITIVITY_DIRS
    )
    parser.add_argument(
        "--direct_comparison", type=Path, default=DEFAULT_DIRECT_COMPARISON
    )
    parser.add_argument(
        "--metabolite_dictionary", type=Path, default=DEFAULT_METABOLITE_DICTIONARY
    )
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--visualization_dir", type=Path, default=DEFAULT_VISUALIZATION_DIR
    )
    parser.add_argument("--high_k", type=int, default=HIGH_K)
    parser.add_argument("--top_edges_per_outcome", type=int, default=100)
    return parser


def run_analysis(args: argparse.Namespace) -> Dict[str, object]:
    if args.high_k < 1 or args.top_edges_per_outcome < 1:
        raise ValueError("high_k and top_edges_per_outcome must be positive")
    specs = _model_specs(
        args.primary_dir, args.replication_dirs, args.sensitivity_dirs
    )
    model_edges, roles, diagnostics = _load_model_edges(specs, args.high_k)
    direct_validation = _validate_against_direct_comparison(
        model_edges, args.direct_comparison
    )
    primary_edges = model_edges["primary"]
    comparison = build_cross_model_comparison(
        model_edges, roles, top_ks=TOP_KS, high_k=args.high_k
    )
    comparison = comparison.merge(
        primary_edges.loc[:, [*EDGE_KEYS, "direction_reversal", "effect_pattern"]],
        on=EDGE_KEYS,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    stability = build_total_stability(model_edges, roles, top_ks=TOP_KS)
    multi = build_multi_outcome_summary(primary_edges, top_ks=TOP_KS)
    dictionary = _load_metabolite_dictionary(args.metabolite_dictionary)

    primary_output = _annotate_metabolites(primary_edges, dictionary)
    comparison_output = _annotate_metabolites(comparison, dictionary)
    top_output = comparison_output.loc[
        comparison_output["primary_total_outcome_rank"] <= args.top_edges_per_outcome
    ].reset_index(drop=True)
    candidate_output = comparison_output.loc[
        comparison_output["effect_pattern"].ne("neither_high")
        | comparison_output["direction_reversal"]
    ].reset_index(drop=True)
    multi_output = _annotate_metabolites(multi, dictionary)
    outputs = {
        "total_effect_primary.csv": primary_output,
        "total_effect_cross_model_comparison.csv": comparison_output,
        "total_effect_top_edges.csv": top_output,
        "direct_vs_total_candidates.csv": candidate_output,
        "total_effect_stability_summary.csv": stability,
        "multi_outcome_total_effect_summary.csv": multi_output,
    }
    for filename, frame in outputs.items():
        _validate_output_frame(frame, filename)
    summary = _summary_payload(
        specs,
        diagnostics,
        direct_validation,
        primary_edges,
        comparison,
        stability,
        args,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in outputs.items():
        frame.to_csv(out_dir / filename, index=False)
    (out_dir / "total_effect_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    render_visualizations(
        primary_output,
        comparison_output,
        stability,
        multi_output,
        args.visualization_dir,
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    run_analysis(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
