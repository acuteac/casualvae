"""Outcome-centered direct-effect screening from exported causal G matrices."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


EDGE_COLUMNS = [
    "modality",
    "outcome",
    "feature",
    "effect",
    "abs_effect",
    "outcome_rank",
]
EDGE_KEYS = ["modality", "outcome", "feature"]
TOP_KS = (20, 50, 100)
MATRIX_FILES = {
    "Protein": "G_protein_to_outcome.csv",
    "Metabolite": "G_metabolite_to_outcome.csv",
}
MODEL_LABEL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METABOLITE_DICTIONARY = (
    PROJECT_ROOT / "artifacts/metadata/metabolite_feature_dictionary.csv"
)
OUTPUT_FILES = (
    "outcome_specific_direct_edges.csv",
    "outcome_specific_cross_model_comparison.csv",
    "outcome_specific_top_edges.csv",
    "outcome_specific_stability_summary.csv",
    "multi_outcome_feature_summary.csv",
    "direct_effect_summary.json",
)


def _validate_top_ks(top_ks: Iterable[int]) -> Tuple[int, ...]:
    values = tuple(sorted({int(value) for value in top_ks}))
    if not values or any(value < 1 for value in values):
        raise ValueError("top_ks must contain positive integers")
    return values


def _validated_matrix(matrix: pd.DataFrame, source: str = "effect matrix") -> pd.DataFrame:
    if matrix.empty:
        raise ValueError(f"{source} is empty")
    labels = [*matrix.index.tolist(), *matrix.columns.tolist()]
    if any(pd.isna(label) or not str(label).strip() for label in labels):
        raise ValueError(f"{source} contains null or blank Outcome/feature labels")
    if matrix.index.has_duplicates:
        raise ValueError(f"{source} has duplicate Outcome rows")
    if matrix.columns.has_duplicates:
        raise ValueError(f"{source} has duplicate feature columns")

    converted = matrix.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(converted.to_numpy(dtype=float)).all():
        raise ValueError(f"{source} contains NaN or infinite effects")
    converted.index = converted.index.map(str)
    converted.columns = converted.columns.map(str)
    if converted.index.has_duplicates:
        raise ValueError(f"{source} has duplicate Outcome labels after conversion")
    if converted.columns.has_duplicates:
        raise ValueError(f"{source} has duplicate feature labels after conversion")
    return converted


def load_effect_matrix(path: Path | str) -> pd.DataFrame:
    """Load an Outcome-by-feature G matrix and reject malformed effects."""
    matrix_path = Path(path)
    if not matrix_path.is_file():
        raise FileNotFoundError(f"Effect matrix not found: {matrix_path}")
    with matrix_path.open("r", encoding="utf-8-sig", newline="") as handle:
        try:
            header = next(csv.reader(handle))
        except StopIteration as exc:
            raise ValueError(f"{matrix_path} is empty") from exc
    feature_headers = [value.strip() for value in header[1:]]
    if not feature_headers or any(not value for value in feature_headers):
        raise ValueError(f"{matrix_path} contains a null or blank feature header")
    if len(feature_headers) != len(set(feature_headers)):
        raise ValueError(f"{matrix_path} contains duplicate feature headers")
    return _validated_matrix(pd.read_csv(matrix_path, index_col=0), str(matrix_path))


def matrix_to_edge_table(matrix: pd.DataFrame, modality: str) -> pd.DataFrame:
    """Convert an Outcome-by-feature matrix to signed Outcome-specific ranks."""
    validated = _validated_matrix(matrix)
    wide = validated.copy()
    wide.index.name = "outcome"
    edges = wide.reset_index().melt(
        id_vars="outcome", var_name="feature", value_name="effect"
    )
    edges.insert(0, "modality", str(modality))
    edges["effect"] = edges["effect"].astype(float)
    edges["abs_effect"] = edges["effect"].abs()
    edges = edges.sort_values(
        ["modality", "outcome", "abs_effect", "feature"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    edges["outcome_rank"] = (
        edges.groupby(["modality", "outcome"], sort=False).cumcount() + 1
    ).astype(int)
    return edges.loc[:, EDGE_COLUMNS]


def _validate_model_label(label: str) -> None:
    if not MODEL_LABEL_PATTERN.fullmatch(label):
        raise ValueError(
            f"Invalid model label {label!r}; use letters, numbers, and underscores"
        )


def _validate_model_edges(
    model_edges: Mapping[str, pd.DataFrame], model_roles: Mapping[str, str]
) -> List[str]:
    if "primary" not in model_edges:
        raise ValueError("model_edges must contain a primary table")
    if set(model_edges) != set(model_roles):
        raise ValueError("model_edges and model_roles must use identical labels")
    other_labels = [label for label in model_edges if label != "primary"]
    if not other_labels:
        raise ValueError("at least one replication or sensitivity model is required")
    for label, edges in model_edges.items():
        _validate_model_label(label)
        missing = sorted(set(EDGE_COLUMNS).difference(edges.columns))
        if missing:
            raise ValueError(f"{label} edge table is missing columns: {missing}")
        if edges.duplicated(EDGE_KEYS).any():
            raise ValueError(f"{label} has duplicate feature-to-Outcome edges")
        if not np.isfinite(
            edges[["effect", "abs_effect", "outcome_rank"]].to_numpy(dtype=float)
        ).all():
            raise ValueError(f"{label} edge table contains non-finite values")
    return other_labels


def build_cross_model_comparison(
    model_edges: Mapping[str, pd.DataFrame],
    model_roles: Mapping[str, str],
    top_ks: Iterable[int] = TOP_KS,
) -> pd.DataFrame:
    """Compare the identical feature-to-Outcome edge across all models."""
    top_ks = _validate_top_ks(top_ks)
    other_labels = _validate_model_edges(model_edges, model_roles)
    labels = ["primary", *other_labels]

    comparison: pd.DataFrame | None = None
    for label in labels:
        selected = model_edges[label].loc[
            :, EDGE_KEYS + ["effect", "abs_effect", "outcome_rank"]
        ].rename(
            columns={
                "effect": f"{label}_effect",
                "abs_effect": f"{label}_abs_effect",
                "outcome_rank": f"{label}_outcome_rank",
            }
        )
        if comparison is None:
            comparison = selected
        else:
            comparison = comparison.merge(
                selected,
                on=EDGE_KEYS,
                how="left",
                validate="one_to_one",
                sort=False,
            )

    assert comparison is not None
    required_model_columns = [
        f"{label}_{field}"
        for label in labels
        for field in ("effect", "abs_effect", "outcome_rank")
    ]
    if comparison[required_model_columns].isna().any().any():
        raise ValueError("one or more models are missing Primary feature-to-Outcome edges")

    primary_sign = np.sign(comparison["primary_effect"].to_numpy(dtype=float))
    agreement_columns: List[str] = []
    for label in other_labels:
        column = f"{label}_sign_agrees_primary"
        comparison[column] = (
            np.sign(comparison[f"{label}_effect"].to_numpy(dtype=float))
            == primary_sign
        )
        agreement_columns.append(column)
    comparison["sign_agreement_count"] = (
        comparison[agreement_columns].sum(axis=1).astype(int)
    )
    comparison["sign_agreement_fraction"] = (
        comparison["sign_agreement_count"] / len(other_labels)
    )

    rank_columns = [f"{label}_outcome_rank" for label in labels]
    for k in top_ks:
        comparison[f"top{k}_support_count"] = (
            comparison[rank_columns].le(k).sum(axis=1).astype(int)
        )
    ranks = comparison[rank_columns].to_numpy(dtype=float)
    comparison["median_outcome_rank"] = np.median(ranks, axis=1)
    comparison["mean_outcome_rank"] = np.mean(ranks, axis=1)
    comparison["best_outcome_rank"] = np.min(ranks, axis=1).astype(int)
    comparison["worst_outcome_rank"] = np.max(ranks, axis=1).astype(int)
    return comparison.sort_values(
        ["modality", "outcome", "primary_outcome_rank", "feature"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_outcome_stability(
    model_edges: Mapping[str, pd.DataFrame],
    model_roles: Mapping[str, str],
    top_ks: Iterable[int] = TOP_KS,
) -> pd.DataFrame:
    """Compute Primary-vs-other stability separately per modality and Outcome."""
    top_ks = _validate_top_ks(top_ks)
    other_labels = _validate_model_edges(model_edges, model_roles)
    primary = model_edges["primary"]
    rows: List[Dict[str, object]] = []
    for (modality, outcome), primary_group in primary.groupby(
        ["modality", "outcome"], sort=True
    ):
        primary_group = primary_group.sort_values("feature", kind="mergesort")
        for label in other_labels:
            other_group = model_edges[label].loc[
                (model_edges[label]["modality"] == modality)
                & (model_edges[label]["outcome"] == outcome)
            ].sort_values("feature", kind="mergesort")
            paired = primary_group.loc[
                :, ["feature", "abs_effect", "outcome_rank"]
            ].merge(
                other_group.loc[:, ["feature", "abs_effect", "outcome_rank"]],
                on="feature",
                how="left",
                suffixes=("_primary", "_other"),
                validate="one_to_one",
                sort=False,
            )
            if len(paired) != len(primary_group) or paired.isna().any().any():
                raise ValueError(
                    f"{label} {modality} {outcome} does not match Primary features"
                )
            correlation = float(
                spearmanr(
                    paired["abs_effect_primary"].to_numpy(dtype=float),
                    paired["abs_effect_other"].to_numpy(dtype=float),
                ).statistic
            )
            if not np.isfinite(correlation):
                raise ValueError(
                    f"Spearman correlation is undefined for {label} {modality} {outcome}"
                )
            row: Dict[str, object] = {
                "modality": modality,
                "outcome": outcome,
                "comparison_model": label,
                "comparison_role": model_roles[label],
                "feature_count": int(len(paired)),
                "spearman_rank_correlation": correlation,
            }
            for k in top_ks:
                primary_top = set(
                    paired.loc[paired["outcome_rank_primary"] <= k, "feature"]
                )
                other_top = set(
                    paired.loc[paired["outcome_rank_other"] <= k, "feature"]
                )
                intersection = len(primary_top & other_top)
                union = len(primary_top | other_top)
                row[f"top{k}_intersection"] = int(intersection)
                row[f"top{k}_jaccard"] = float(intersection / union if union else 1.0)
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["modality", "outcome", "comparison_model"], kind="mergesort"
    ).reset_index(drop=True)


def build_multi_outcome_summary(
    primary_edges: pd.DataFrame, top_ks: Iterable[int] = TOP_KS
) -> pd.DataFrame:
    """Count Outcomes in which each feature appears in a Primary Top-K list."""
    top_ks = _validate_top_ks(top_ks)
    rows: List[Dict[str, object]] = []
    for (modality, feature), group in primary_edges.groupby(
        ["modality", "feature"], sort=True
    ):
        row: Dict[str, object] = {"modality": modality, "feature": feature}
        for k in top_ks:
            outcomes = sorted(
                group.loc[group["outcome_rank"] <= k, "outcome"].astype(str).tolist()
            )
            row[f"n_outcomes_top{k}"] = int(len(outcomes))
            row[f"outcomes_top{k}"] = ";".join(outcomes) if outcomes else "none"
        rows.append(row)
    sort_columns = [
        "modality",
        *[f"n_outcomes_top{k}" for k in top_ks],
        "feature",
    ]
    ascending = [True, *([False] * len(top_ks)), True]
    return pd.DataFrame(rows).sort_values(
        sort_columns, ascending=ascending, kind="mergesort"
    ).reset_index(drop=True)


def load_metabolite_dictionary(path: Path | str) -> pd.DataFrame:
    dictionary_path = Path(path)
    if not dictionary_path.is_file():
        raise FileNotFoundError(f"Metabolite dictionary not found: {dictionary_path}")
    dictionary = pd.read_csv(dictionary_path, dtype=str, keep_default_na=False)
    required = ["feature_code", "metabolite_name"]
    missing = [column for column in required if column not in dictionary.columns]
    if missing:
        raise ValueError(f"Metabolite dictionary is missing columns: {missing}")
    dictionary = dictionary.loc[:, required].copy()
    if dictionary["feature_code"].duplicated().any():
        raise ValueError("Metabolite dictionary feature_code values must be unique")
    if dictionary[required].eq("").any().any():
        raise ValueError("Metabolite dictionary contains blank codes or names")
    return dictionary


def add_metabolite_annotations(
    frame: pd.DataFrame, dictionary: pd.DataFrame
) -> pd.DataFrame:
    """Add names while retaining feature code as the analysis key."""
    annotated = frame.copy()
    annotated["feature_code"] = annotated["feature"].astype(str)
    annotated["metabolite_name"] = "not_applicable"
    metabolite_mask = annotated["modality"].eq("Metabolite")
    name_map = dictionary.set_index("feature_code")["metabolite_name"]
    mapped = annotated.loc[metabolite_mask, "feature_code"].map(name_map)
    if mapped.isna().any():
        missing = sorted(
            annotated.loc[metabolite_mask & mapped.isna(), "feature_code"].unique()
        )
        raise ValueError(f"Metabolite dictionary has no mapping for: {missing[:20]}")
    annotated.loc[metabolite_mask, "metabolite_name"] = mapped.to_numpy()
    return annotated


def _derive_model_label(directory: Path, role: str, index: int) -> str:
    parts = [part.lower() for part in directory.parts]
    match = re.search(r"(?:model_)?seed[_-]?(\d+)$", directory.name.lower())
    seed = match.group(1) if match else None
    if seed and "variable_split" in parts:
        return f"variable{seed}"
    if seed and "fixed_split" in parts:
        return f"fixed{seed}"
    if seed:
        return f"{role}{seed}"
    return f"{role}{index}"


def _model_specs(
    primary_dir: Path,
    replication_dirs: Sequence[Path],
    sensitivity_dirs: Sequence[Path],
) -> List[Dict[str, object]]:
    if len(replication_dirs) != 2:
        raise ValueError("4.2 requires exactly two replication directories")
    if len(sensitivity_dirs) != 1:
        raise ValueError("4.2 requires exactly one sensitivity directory")
    all_directories = [primary_dir, *replication_dirs, *sensitivity_dirs]
    resolved = [str(Path(path).resolve(strict=False)).casefold() for path in all_directories]
    if len(resolved) != len(set(resolved)):
        raise ValueError("Primary, replication, and sensitivity must use distinct directories")
    specs: List[Dict[str, object]] = [
        {"label": "primary", "role": "primary", "directory": primary_dir}
    ]
    for index, directory in enumerate(replication_dirs, start=1):
        specs.append(
            {
                "label": _derive_model_label(directory, "replication", index),
                "role": "replication",
                "directory": directory,
            }
        )
    for index, directory in enumerate(sensitivity_dirs, start=1):
        specs.append(
            {
                "label": _derive_model_label(directory, "sensitivity", index),
                "role": "sensitivity",
                "directory": directory,
            }
        )
    labels = [str(spec["label"]) for spec in specs]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Derived model labels are not unique: {labels}")
    for label in labels:
        _validate_model_label(label)
    return specs


def _load_all_models(
    specs: Sequence[Mapping[str, object]],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    model_edges: Dict[str, pd.DataFrame] = {}
    roles: Dict[str, str] = {}
    primary_schemas: Dict[str, Tuple[List[str], List[str]]] = {}
    for spec in specs:
        label = str(spec["label"])
        role = str(spec["role"])
        directory = Path(spec["directory"])
        modality_edges: List[pd.DataFrame] = []
        for modality, filename in MATRIX_FILES.items():
            matrix = load_effect_matrix(directory / filename)
            if label == "primary":
                primary_schemas[modality] = (
                    matrix.index.tolist(),
                    matrix.columns.tolist(),
                )
            else:
                expected_outcomes, expected_features = primary_schemas[modality]
                if set(matrix.index) != set(expected_outcomes):
                    raise ValueError(f"{label} {modality} Outcomes differ from Primary")
                if set(matrix.columns) != set(expected_features):
                    raise ValueError(f"{label} {modality} features differ from Primary")
                matrix = matrix.reindex(
                    index=expected_outcomes, columns=expected_features
                )
            modality_edges.append(matrix_to_edge_table(matrix, modality))
        model_edges[label] = pd.concat(modality_edges, ignore_index=True)
        roles[label] = role
    return model_edges, roles


def _validate_output_frame(frame: pd.DataFrame, label: str) -> None:
    if frame.empty:
        raise ValueError(f"{label} is empty")
    if frame.isna().any().any():
        raise ValueError(f"{label} contains missing values")
    numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{label} contains NaN or infinite numeric values")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def _aggregate_stability(stability: pd.DataFrame) -> List[Dict[str, object]]:
    metric_columns = [
        "spearman_rank_correlation",
        *[f"top{k}_jaccard" for k in TOP_KS],
    ]
    aggregate = (
        stability.groupby(["modality", "outcome"], sort=True)[metric_columns]
        .mean()
        .add_prefix("mean_")
        .reset_index()
    )
    return aggregate.to_dict(orient="records")


def _summary_payload(
    specs: Sequence[Mapping[str, object]],
    primary_edges: pd.DataFrame,
    comparison: pd.DataFrame,
    stability: pd.DataFrame,
    args: argparse.Namespace,
) -> Dict[str, object]:
    models: Dict[str, object] = {"replication": [], "sensitivity": []}
    for spec in specs:
        item = {
            "label": str(spec["label"]),
            "role": str(spec["role"]),
            "directory": str(Path(spec["directory"]).resolve()),
        }
        if spec["role"] == "primary":
            models["primary"] = item
        else:
            models[str(spec["role"])].append(item)
    feature_counts = {
        modality: int(
            primary_edges.loc[
                primary_edges["modality"] == modality, "feature"
            ].nunique()
        )
        for modality in MATRIX_FILES
    }
    other_model_count = len(specs) - 1
    total_model_count = len(specs)
    return {
        "analysis": "4.2 Outcome-centered Direct Effect Candidate Screening",
        "analysis_unit": "feature_to_outcome_edge",
        "matrix_orientation": (
            "rows=Outcome, columns=feature, values=feature_to_Outcome_direct_effect"
        ),
        "models": models,
        "outcomes": sorted(primary_edges["outcome"].unique().tolist()),
        "outcome_count": int(primary_edges["outcome"].nunique()),
        "feature_counts": feature_counts,
        "primary_edge_count": int(len(primary_edges)),
        "cross_model_edge_count": int(len(comparison)),
        "stability_comparison_count": int(len(stability)),
        "top_k_thresholds": list(TOP_KS),
        "top_edges_per_outcome": int(args.top_edges_per_outcome),
        "ranking_rule": [
            "within each modality and Outcome",
            "abs(effect) descending",
            "feature ascending for deterministic ties",
        ],
        "sign_agreement_definition": (
            "number of other models whose signed effect matches Primary; "
            f"denominator={other_model_count}"
        ),
        "top_k_support_definition": (
            f"number of {total_model_count} models in which the same edge is "
            "within that Outcome Top-K"
        ),
        "metabolite_dictionary": str(Path(args.metabolite_dictionary).resolve()),
        "outcome_stability_mean_across_other_models": _aggregate_stability(stability),
        "output_files": list(OUTPUT_FILES),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Outcome-centered direct-effect screening from existing "
            "Protein/Metabolite-to-Outcome G matrices without retraining."
        )
    )
    parser.add_argument("--primary_dir", type=Path, required=True)
    parser.add_argument("--replication_dirs", type=Path, nargs=2, required=True)
    parser.add_argument("--sensitivity_dirs", type=Path, nargs=1, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--metabolite_dictionary",
        type=Path,
        default=DEFAULT_METABOLITE_DICTIONARY,
    )
    parser.add_argument("--top_edges_per_outcome", type=int, default=100)
    return parser


def run_screening(args: argparse.Namespace) -> Dict[str, object]:
    if args.top_edges_per_outcome < 1:
        raise ValueError("top_edges_per_outcome must be positive")
    specs = _model_specs(
        args.primary_dir, args.replication_dirs, args.sensitivity_dirs
    )
    model_edges, roles = _load_all_models(specs)
    primary_edges = model_edges["primary"]
    comparison = build_cross_model_comparison(model_edges, roles)
    stability = build_outcome_stability(model_edges, roles)
    multi_outcome = build_multi_outcome_summary(primary_edges)
    dictionary = load_metabolite_dictionary(args.metabolite_dictionary)

    direct_output = add_metabolite_annotations(primary_edges, dictionary)
    comparison_output = add_metabolite_annotations(comparison, dictionary)
    top_output = comparison_output.loc[
        comparison_output["primary_outcome_rank"] <= args.top_edges_per_outcome
    ].reset_index(drop=True)
    multi_output = add_metabolite_annotations(multi_outcome, dictionary)

    outputs = {
        "outcome_specific_direct_edges.csv": direct_output,
        "outcome_specific_cross_model_comparison.csv": comparison_output,
        "outcome_specific_top_edges.csv": top_output,
        "outcome_specific_stability_summary.csv": stability,
        "multi_outcome_feature_summary.csv": multi_output,
    }
    for label, frame in outputs.items():
        _validate_output_frame(frame, label)

    summary = _summary_payload(
        specs, primary_edges, comparison, stability, args
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in outputs.items():
        _write_csv(frame, out_dir / filename)
    (out_dir / "direct_effect_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    run_screening(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
