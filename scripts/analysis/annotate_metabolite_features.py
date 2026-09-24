"""Annotate encoded UK Biobank Nightingale metabolite features in 4.2 outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from pandas.testing import assert_frame_equal


FEATURE_PATTERN = re.compile(r"^p(?P<field_id>\d+)_i(?P<instance>\d+)$")
ANNOTATION_COLUMNS = [
    "feature_code",
    "field_id",
    "instance",
    "metabolite_name",
    "metabolite_name_short",
    "annotation_source",
    "annotation_status",
    "category",
    "subclass",
    "ukb_field_title",
]
OFFICIAL_RESOURCE_URL = (
    "https://biobank.ndph.ox.ac.uk/ukb/ukb/docs/"
    "Nightingale_biomarker_groups.txt"
)


def parse_feature_code(feature_code: object) -> Tuple[int | None, int | None]:
    match = FEATURE_PATTERN.fullmatch(str(feature_code))
    if not match:
        return None, None
    return int(match.group("field_id")), int(match.group("instance"))


def _find_column(frame: pd.DataFrame, names: Iterable[str], label: str) -> str:
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    for name in names:
        if name.lower() in normalized:
            return normalized[name.lower()]
    raise ValueError(
        f"Mapping source has no {label} column; available columns: {list(frame.columns)}"
    )


def load_mapping_source(path: Path | str) -> pd.DataFrame:
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Mapping source not found: {source_path}")
    separator = "\t" if source_path.suffix.lower() in {".tsv", ".txt"} else ","
    mapping = pd.read_csv(source_path, sep=separator, dtype=str, keep_default_na=False)
    field_column = _find_column(mapping, ("field_id", "field id"), "field_id")
    title_column = _find_column(
        mapping, ("title", "description", "metabolite_name"), "title"
    )
    group_column = next(
        (
            column
            for column in mapping.columns
            if str(column).strip().lower() in {"group", "category"}
        ),
        None,
    )
    subgroup_column = next(
        (
            column
            for column in mapping.columns
            if str(column).strip().lower() in {"subgroup", "subclass"}
        ),
        None,
    )
    normalized = pd.DataFrame(
        {
            "field_id": pd.to_numeric(mapping[field_column], errors="coerce"),
            "title": mapping[title_column].astype(str).str.strip(),
            "Group": (
                mapping[group_column].astype(str).str.strip()
                if group_column is not None
                else ""
            ),
            "Subgroup": (
                mapping[subgroup_column].astype(str).str.strip()
                if subgroup_column is not None
                else ""
            ),
        }
    )
    invalid = normalized["field_id"].isna() | normalized["title"].eq("")
    if invalid.any():
        bad_rows = mapping.index[invalid].tolist()[:10]
        raise ValueError(f"Mapping source has invalid field/title rows: {bad_rows}")
    normalized["field_id"] = normalized["field_id"].astype(int)
    return normalized


def build_feature_dictionary(
    feature_codes: Sequence[object],
    mapping: pd.DataFrame,
    annotation_source: str,
) -> pd.DataFrame:
    codes = [str(code) for code in feature_codes]
    if len(codes) != len(set(codes)):
        raise ValueError("Metabolite feature codes must be unique")
    required = {"field_id", "title", "Group", "Subgroup"}
    missing = sorted(required.difference(mapping.columns))
    if missing:
        raise ValueError(f"Normalized mapping is missing columns: {missing}")

    source_by_field: Dict[int, Dict[str, str]] = {}
    ambiguous_fields = set()
    for field_id, group in mapping.groupby("field_id", sort=True):
        titles = sorted({str(value).strip() for value in group["title"] if str(value).strip()})
        if len(titles) != 1:
            ambiguous_fields.add(int(field_id))
            continue
        categories = sorted(
            {str(value).strip() for value in group["Group"] if str(value).strip()}
        )
        subclasses = sorted(
            {str(value).strip() for value in group["Subgroup"] if str(value).strip()}
        )
        source_by_field[int(field_id)] = {
            "title": titles[0],
            "category": categories[0] if len(categories) == 1 else "",
            "subclass": subclasses[0] if len(subclasses) == 1 else "",
        }

    rows: List[Dict[str, object]] = []
    for feature_code in codes:
        field_id, instance = parse_feature_code(feature_code)
        if field_id in ambiguous_fields:
            status = "ambiguous"
            details = None
        else:
            details = source_by_field.get(field_id) if field_id is not None else None
            status = "mapped" if details is not None else "unmapped"
        name = details["title"] if details is not None else feature_code
        rows.append(
            {
                "feature_code": feature_code,
                "field_id": field_id,
                "instance": instance,
                "metabolite_name": name,
                "metabolite_name_short": name,
                "annotation_source": str(annotation_source),
                "annotation_status": status,
                "category": details["category"] if details is not None else "",
                "subclass": details["subclass"] if details is not None else "",
                "ukb_field_title": details["title"] if details is not None else "",
            }
        )
    dictionary = pd.DataFrame(rows, columns=ANNOTATION_COLUMNS)
    dictionary["field_id"] = pd.array(dictionary["field_id"], dtype="Int64")
    dictionary["instance"] = pd.array(dictionary["instance"], dtype="Int64")
    return dictionary.sort_values(
        ["field_id", "instance", "feature_code"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def annotate_table(
    original: pd.DataFrame,
    dictionary: pd.DataFrame,
    feature_column: str = "feature",
) -> pd.DataFrame:
    if feature_column not in original.columns:
        raise ValueError(f"Table has no {feature_column!r} column")
    if dictionary["feature_code"].duplicated().any():
        raise ValueError("Dictionary feature_code values must be unique")
    annotated = original.copy()
    row_order_column = "_annotation_row_order"
    join_column = "_annotation_feature_key"
    while row_order_column in annotated.columns:
        row_order_column = f"_{row_order_column}"
    while join_column in annotated.columns:
        join_column = f"_{join_column}"
    annotated[row_order_column] = range(len(annotated))
    annotated[join_column] = annotated[feature_column]
    if "modality" in annotated.columns:
        is_metabolite = (
            annotated["modality"].astype(str).str.strip().str.lower().eq("metabolite")
        )
        annotated.loc[~is_metabolite, join_column] = pd.NA
    annotated = annotated.merge(
        dictionary.loc[:, ANNOTATION_COLUMNS],
        how="left",
        left_on=join_column,
        right_on="feature_code",
        validate="many_to_one",
        sort=False,
    ).sort_values(row_order_column, kind="mergesort")
    annotated = annotated.drop(columns=[row_order_column, join_column]).reset_index(
        drop=True
    )
    assert_frame_equal(
        annotated.loc[:, original.columns],
        original.reset_index(drop=True),
        check_exact=True,
        check_dtype=True,
    )
    return annotated


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path_key(path: Path) -> str:
    return str(path.resolve(strict=False)).casefold()


def _validate_path_collisions(
    input_paths: Dict[str, Path], output_paths: Dict[str, Path]
) -> None:
    input_by_path = {_path_key(path): label for label, path in input_paths.items()}
    output_by_path: Dict[str, str] = {}
    for label, path in output_paths.items():
        key = _path_key(path)
        if key in input_by_path:
            raise ValueError(
                f"Output {label} collides with an input ({input_by_path[key]}): {path}"
            )
        if key in output_by_path:
            raise ValueError(
                f"Output {label} collides with another output "
                f"({output_by_path[key]}): {path}"
            )
        output_by_path[key] = label


def _build_candidate_summary(annotated_tiers: pd.DataFrame) -> pd.DataFrame:
    required = {
        "primary_direct_rank",
        "feature_code",
        "metabolite_name",
        "primary_direct_score",
        "strongest_outcome",
        "strongest_effect",
        "candidate_tier",
        "sign_agreement_count",
        "sign_agreement_fraction",
        "all_edge_sign_agreement_fraction",
        "annotation_status",
    }
    missing = sorted(required.difference(annotated_tiers.columns))
    if missing:
        raise ValueError(f"Candidate tier table is missing columns: {missing}")
    return annotated_tiers.loc[:, list(required)].rename(
        columns={
            "primary_direct_rank": "primary_rank",
            "primary_direct_score": "DirectScore",
            "candidate_tier": "tier",
        }
    ).loc[
        :,
        [
            "primary_rank",
            "feature_code",
            "metabolite_name",
            "DirectScore",
            "strongest_outcome",
            "strongest_effect",
            "tier",
            "sign_agreement_count",
            "sign_agreement_fraction",
            "all_edge_sign_agreement_fraction",
            "annotation_status",
        ],
    ].sort_values(["primary_rank", "feature_code"], kind="mergesort").reset_index(drop=True)


def _summary_payload(
    dictionary: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    mapping_source: Path,
) -> Dict[str, object]:
    counts = dictionary["annotation_status"].value_counts()
    total = int(len(dictionary))
    mapped = int(counts.get("mapped", 0))
    top10 = candidate_summary.nsmallest(10, "primary_rank")
    return {
        "total_metabolite_features": total,
        "mapped_count": mapped,
        "unmapped_count": int(counts.get("unmapped", 0)),
        "ambiguous_count": int(counts.get("ambiguous", 0)),
        "mapping_rate": mapped / total if total else 0.0,
        "annotation_sources": sorted(
            dictionary["annotation_source"].dropna().astype(str).unique().tolist()
        ),
        "mapping_source_file": str(mapping_source.resolve()),
        "mapping_source_sha256": _source_sha256(mapping_source),
        "unmapped_features": dictionary.loc[
            dictionary["annotation_status"] == "unmapped", "feature_code"
        ].tolist(),
        "ambiguous_features": dictionary.loc[
            dictionary["annotation_status"] == "ambiguous", "feature_code"
        ].tolist(),
        "top10_primary_metabolites": [
            {
                "rank": int(row.primary_rank),
                "feature_code": str(row.feature_code),
                "metabolite_name": str(row.metabolite_name),
                "DirectScore": float(row.DirectScore),
                "strongest_outcome": str(row.strongest_outcome),
                "tier": str(row.tier),
            }
            for row in top10.itertuples(index=False)
        ],
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Map encoded UK Biobank Nightingale metabolite features to official "
            "names and add annotations to existing 4.2 outputs."
        )
    )
    parser.add_argument("--screening_dir", type=Path, required=True)
    parser.add_argument("--mapping_source", type=Path, required=True)
    parser.add_argument("--dictionary_out", type=Path, required=True)
    parser.add_argument(
        "--annotation_source", default=f"UK Biobank Data Showcase Resource 3543: {OFFICIAL_RESOURCE_URL}"
    )
    return parser


def run_annotation(args: argparse.Namespace) -> Dict[str, object]:
    screening_dir = Path(args.screening_dir)
    ranking_path = screening_dir / "metabolite_direct_ranking.csv"
    tiers_path = screening_dir / "metabolite_candidate_tiers.csv"
    mapping_source = Path(args.mapping_source)
    dictionary_out = Path(args.dictionary_out)
    optional_files = {
        "cross_model_direct_effect_comparison.csv":
            "cross_model_direct_effect_comparison_annotated.csv",
        "top_direct_edges_by_outcome.csv": "top_direct_edges_by_outcome_annotated.csv",
    }
    optional_inputs = {
        input_name: screening_dir / input_name
        for input_name in optional_files
        if (screening_dir / input_name).is_file()
    }
    input_paths = {
        "mapping_source": mapping_source,
        "metabolite_direct_ranking": ranking_path,
        "metabolite_candidate_tiers": tiers_path,
        **optional_inputs,
    }
    output_paths = {
        "dictionary_out": dictionary_out,
        "metabolite_direct_ranking_annotated":
            screening_dir / "metabolite_direct_ranking_annotated.csv",
        "metabolite_candidate_tiers_annotated":
            screening_dir / "metabolite_candidate_tiers_annotated.csv",
        "metabolite_candidate_annotation_summary":
            screening_dir / "metabolite_candidate_annotation_summary.csv",
        "metabolite_annotation_summary":
            screening_dir / "metabolite_annotation_summary.json",
        **{
            output_name: screening_dir / output_name
            for input_name, output_name in optional_files.items()
            if input_name in optional_inputs
        },
    }
    _validate_path_collisions(input_paths, output_paths)

    ranking = pd.read_csv(ranking_path)
    tiers = pd.read_csv(tiers_path)
    if "feature" not in ranking.columns:
        raise ValueError(f"{ranking_path} has no feature column")
    if "feature" not in tiers.columns:
        raise ValueError(f"{tiers_path} has no feature column")
    if set(ranking["feature"]) != set(tiers["feature"]):
        raise ValueError("Ranking and tier tables contain different metabolite features")

    mapping = load_mapping_source(mapping_source)
    dictionary = build_feature_dictionary(
        ranking["feature"].tolist(), mapping, args.annotation_source
    )
    annotated_ranking = annotate_table(ranking, dictionary)
    annotated_tiers = annotate_table(tiers, dictionary)
    annotated_optional: Dict[str, pd.DataFrame] = {}
    for input_name, output_name in optional_files.items():
        input_path = screening_dir / input_name
        if input_path.is_file():
            original = pd.read_csv(input_path, low_memory=False)
            annotated_optional[output_name] = annotate_table(
                original, dictionary
            )

    candidate_summary = _build_candidate_summary(annotated_tiers)
    summary = _summary_payload(dictionary, candidate_summary, mapping_source)

    dictionary_out.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(dictionary, dictionary_out)
    _write_csv(
        annotated_ranking, screening_dir / "metabolite_direct_ranking_annotated.csv"
    )
    _write_csv(
        annotated_tiers, screening_dir / "metabolite_candidate_tiers_annotated.csv"
    )
    for output_name, annotated in annotated_optional.items():
        _write_csv(annotated, screening_dir / output_name)
    _write_csv(
        candidate_summary,
        screening_dir / "metabolite_candidate_annotation_summary.csv",
    )
    (screening_dir / "metabolite_annotation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    run_annotation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
