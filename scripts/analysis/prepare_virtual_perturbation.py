"""4.4.1: prepare input-feature perturbation plans; never load or execute a model.

Original feature codes identify molecules. Biological annotations never enter ranking.
The saved plans are input-SD shifts, not claims of strict SCM do interventions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis import direct_effect_screening as direct
from scripts.analysis import total_effect_propagation as total

ROOT = Path(__file__).resolve().parents[2]
KEYS = total.EDGE_KEYS
MODALITIES = {"Protein": "protein", "Metabolite": "metabolite"}
SPLITS = [f"{m}_{s}_eids" for m in ("protein", "metabolite", "paired") for s in ("train", "val")]
CONFIG_KEYS = ["anchor_cols", "phenotype_continuous_cols", "phenotype_binary_cols",
               "protein_cols", "metabolite_cols", "shared_dims", "private_dims", "hidden_dims",
               "protein_availability_threshold", "metabolite_availability_threshold",
               "outcome_exogenous_mode", "outcome_exogenous_sigma"]
SORT_COLUMNS = ["primary_top20_outcome_count", "primary_top50_outcome_count",
                "primary_mean_total_rank", "high_rank_mean_top20_model_support",
                "mean_total_sign_agreement", "feature"]
SORT_ASCENDING = [False, False, True, False, False, True]
CANDIDATE_COLUMNS = ["modality", "feature", "feature_display", "candidate_role", "rank",
    "primary_top20_outcome_count", "primary_top50_outcome_count", "primary_mean_total_rank",
    "primary_best_total_rank", "high_rank_mean_top20_model_support", "mean_total_sign_agreement",
    "primary_direct_top20_outcome_count", "primary_mean_direct_rank", "mean_abs_direct_effect",
    "mean_abs_total_effect", "mean_abs_indirect_effect", "direction_reversal_outcome_count",
    "best_total_rank_percentile", "mean_total_rank_percentile", "selection_reason"]
COHORT_COLUMNS = ["model", "model_role", "eid", "modality", "feature", "candidate_role",
    "protein_observed_fraction", "metabolite_observed_fraction", "protein_available", "metabolite_available",
    "paired_available", "candidate_observed", "candidate_missing", "baseline_value",
    "protein_train", "protein_val", "metabolite_train", "metabolite_val", "paired_train", "paired_val",
    "not_joint_trained", "strictly_untrained", "cohort_type", "eligible_main", "eligible_exploratory",
    "eligible", "exclusion_reason"]
PLAN_COLUMNS = ["model", "model_role", "eid", "modality", "feature", "candidate_role", "cohort_type",
    "strictly_untrained", "baseline_value", "dose_sd", "target_value", "is_baseline",
    "support_lower", "support_upper", "support_n", "out_of_support", "executable", "nonexecution_reason"]
EFFECT_FIELDS = ["direct_effect", "indirect_effect", "total_effect", "direct_outcome_rank",
                 "total_outcome_rank", "rank_shift"]
PHASE2_REQUIREMENTS = {
    "encoder_sampling": "Encoder reparameterization samples in eval mode. Define paired deterministic latent means or controlled common-random-number sampling before phase 2; do not change forward in phase 1.",
    "scm_and_outcome_noise": "Use recorded best_tau, deterministic SCM gates and an explicit outcome-exogenous-noise policy; pair baseline/dose randomness.",
    "weights": "Phase 1 hashes nonempty model.pt bytes without deserializing. Phase 2 must strictly validate checkpoint keys/shapes against recorded architecture before inference.",
    "execution_contract": "Require preflight status=ready, verify manifest/config hashes and schema_version, execute only executable=true rows with their same-model/eid/feature zero-SD baseline. Keep exploratory cohorts separate.",
}


class PreflightError(ValueError):
    """Fatal input errors, also written to preflight_report.json."""


def _json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _git(*command):
    try:
        result = subprocess.run(["git", "-C", str(ROOT), *command], capture_output=True, text=True)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_csv(path, required=()):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    if not header or len(header) != len(set(header)) or any(not c.strip() for c in header):
        raise ValueError(f"{path}: blank or duplicate column/feature codes")
    frame = pd.read_csv(path, dtype={c: str for c in [*KEYS, "eid"] if c in header})
    missing = sorted(set(required) - set(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"{path}: empty table or missing columns: {missing}")
    if frame[list(required)].isna().any().any():
        raise ValueError(f"{path}: missing key values")
    return frame


def _finite(frame, columns, source):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing columns: {missing}")
    values = frame[list(columns)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{source}: non-finite effect/rank/metadata values")


def _same_values(actual, expected, keys, columns, source):
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():
        raise ValueError(f"{source}: duplicate feature/edge keys")
    a = actual.set_index(keys).sort_index()
    e = expected.set_index(keys).sort_index()
    if not a.index.equals(e.index):
        raise ValueError(f"{source}: feature/Outcome keys mismatch")
    for col in columns:
        if col not in a or col not in e:
            raise ValueError(f"{source}: missing column {col}")
        if pd.api.types.is_numeric_dtype(e[col]):
            _finite(a, [col], source)
            matches = np.allclose(a[col].to_numpy(dtype=float), e[col].to_numpy(dtype=float), rtol=1e-7, atol=1e-10)
        else:
            matches = a[col].astype(str).equals(e[col].astype(str))
        if not matches:
            raise ValueError(f"{source}: inconsistent {col}")


def _validate_config(config):
    features = []
    for key in CONFIG_KEYS[:5]:
        cols = config.get(key)
        if not isinstance(cols, list) or (key != "anchor_cols" and not cols):
            raise ValueError(f"config: {key} must be a feature list")
        if any(not isinstance(c, str) or not c.strip() or c == "eid" for c in cols):
            raise ValueError(f"config: invalid feature code in {key}")
        if len(cols) != len(set(cols)):
            raise ValueError(f"config: duplicate feature codes in {key}")
        features.extend(cols)
    if len(features) != len(set(features)) or any(c == "index" or c.startswith("Unnamed") for c in features):
        raise ValueError("config: overlapping modality/Outcome features or artifact columns")
    if len(config["phenotype_continuous_cols"]) != 14 or len(config["phenotype_binary_cols"]) != 3:
        raise ValueError("config: expected all 14 continuous and 3 binary Outcomes")
    for name in MODALITIES.values():
        threshold = float(config.get(f"{name}_availability_threshold", .5))
        if not np.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError(f"config: invalid {name} availability threshold")
    if config.get("outcome_exogenous_mode", "zero") not in ("zero", "gaussian"):
        raise ValueError("config: unsupported outcome_exogenous_mode")
    for key, blocks in [("shared_dims", ("protein", "metabolite", "outcome")),
                        ("private_dims", ("protein", "metabolite"))]:
        dims = config.get(key)
        if not isinstance(dims, dict) or set(dims) != set(blocks):
            raise ValueError(f"config: {key} must map exactly {blocks} to dimensions")
        minimum = 1 if key == "shared_dims" else 0
        if any(type(dims[b]) is not int or dims[b] < minimum for b in blocks):
            raise ValueError(f"config: {key} dimensions must be integers >= {minimum}")
    hidden = config.get("hidden_dims")
    if not isinstance(hidden, dict):
        raise ValueError("config: hidden_dims must be an object")
    for name in MODALITIES.values():
        layers = hidden.get(name)
        if not isinstance(layers, list) or any(type(n) is not int or n <= 0 for n in layers):
            raise ValueError(f"config: hidden_dims.{name} must be a list of positive integers")
    sigma = config.get("outcome_exogenous_sigma", 1.)
    if type(sigma) not in (float, int) or not np.isfinite(sigma) or sigma < 0:
        raise ValueError("config: outcome_exogenous_sigma must be finite and nonnegative")


def _active_config_value(config, key):
    value = config.get(key)
    # The current constructor consumes hidden_dims.protein/metabolite only.
    # Preserve the raw config in provenance and warn about historical extras.
    if key == "hidden_dims" and isinstance(value, dict):
        return {k: v for k, v in value.items() if k != "outcome"}
    defaults = {"protein_availability_threshold": .5, "metabolite_availability_threshold": .5,
                "outcome_exogenous_mode": "zero", "outcome_exogenous_sigma": 1.}
    return config.get(key, defaults.get(key))


def load_effects(args, config, specs):
    """Verify actual 4.2/4.3 columns, recomputing only ranks/support summaries."""
    p = _read_csv(args.total_dir / "total_effect_primary.csv", KEYS)
    c = _read_csv(args.total_dir / "total_effect_cross_model_comparison.csv", KEYS)
    d = _read_csv(args.direct_dir / "outcome_specific_cross_model_comparison.csv", KEYS)
    outcomes = config["phenotype_continuous_cols"] + config["phenotype_binary_cols"]
    expected_keys = pd.DataFrame([(m, o, f) for m, name in MODALITIES.items()
                                  for o in outcomes for f in config[f"{name}_cols"]], columns=KEYS)
    _same_values(p, expected_keys, KEYS, [], "Primary/config modality and features")
    _finite(p, EFFECT_FIELDS, "Primary")
    for field in ("direct_outcome_rank", "total_outcome_rank"):
        if not np.equal(p[field], np.floor(p[field])).all() or p[field].lt(1).any():
            raise ValueError(f"Primary: {field} must contain positive integers")
    if not np.allclose(p.total_effect, p.direct_effect + p.indirect_effect, rtol=1e-7, atol=1e-10):
        raise ValueError("Primary: direct + indirect != total")
    models = {}
    roles = {s["label"]: s["role"] for s in specs}
    for label in roles:
        fields = [f"{label}_{f}" for f in EFFECT_FIELDS]
        _finite(c, fields, label)
        edges = c[[*KEYS, *fields]].rename(columns={f"{label}_{f}": f for f in EFFECT_FIELDS})
        _same_values(edges, p, KEYS, [], label)
        if not np.allclose(edges.total_effect, edges.direct_effect + edges.indirect_effect, rtol=1e-7, atol=1e-10):
            raise ValueError(f"{label}: direct + indirect != total")
        for effect in ("direct", "total"):
            ranked = total._rank_effect_column(edges, f"{effect}_effect")
            if not np.array_equal(ranked.to_numpy(), edges[f"{effect}_outcome_rank"].to_numpy()):
                raise ValueError(f"{label}: {effect} ranks disagree with per-modality/Outcome abs-effect ordering")
        if not np.array_equal(edges.rank_shift, edges.direct_outcome_rank - edges.total_outcome_rank):
            raise ValueError(f"{label}: rank_shift mismatch")
        edges["direction_reversal"] = edges.direct_effect * edges.total_effect < 0
        edges["effect_pattern"] = "not_used"
        models[label] = edges
    _same_values(p, models["primary"], KEYS, EFFECT_FIELDS, "Primary vs comparison")
    rebuilt = total.build_cross_model_comparison(models, roles)
    checks = [f"total_top{k}_support_count" for k in total.TOP_KS] + ["total_sign_agreement_count", "total_sign_agreement_fraction"]
    _same_values(c, rebuilt, KEYS, checks, "4.3 cross-model support")
    total._validate_against_direct_comparison(models, args.direct_dir / "outcome_specific_cross_model_comparison.csv")
    direct_models = {label: edges[[*KEYS, "direct_effect", "direct_outcome_rank"]].rename(
        columns={"direct_effect": "effect", "direct_outcome_rank": "outcome_rank"}).assign(
            abs_effect=edges.direct_effect.abs()) for label, edges in models.items()}
    rebuilt_d = direct.build_cross_model_comparison(direct_models, roles)
    direct_numeric = [f"{label}_{field}" for label in roles for field in ("effect", "abs_effect", "outcome_rank")]
    _same_values(d, rebuilt_d, KEYS, [*direct_numeric, "sign_agreement_count", "sign_agreement_fraction", "top20_support_count",
                                   "top50_support_count", "top100_support_count"], "4.2 cross-model support")
    for path, summary in [
        (args.total_dir / "multi_outcome_total_effect_summary.csv", total.build_multi_outcome_summary(p)),
        (args.direct_dir / "multi_outcome_feature_summary.csv", direct.build_multi_outcome_summary(direct_models["primary"]))]:
        _same_values(_read_csv(path, ["modality", "feature"]), summary, ["modality", "feature"],
                     [col for col in summary if col.startswith("n_outcomes") or col.startswith("outcomes_top") or col.startswith("outcomes_total")], str(path))
    # Display annotations are already attached to real exports. No dictionary is
    # needed for ranking; insist that all supplied annotations agree by code.
    names = {}
    for frame in (p, c, d):
        if "metabolite_name" not in frame:
            raise ValueError("4.2/4.3: missing metabolite_name display column")
        for feature, group in frame[frame.modality == "Metabolite"].groupby("feature"):
            values = group.metabolite_name.dropna().unique().tolist()
            if len(values) != 1 or not str(values[0]).strip() or values[0] == "not_applicable":
                raise ValueError(f"ambiguous or missing metabolite display name: {feature}")
            if feature in names and names[feature] != values[0]:
                raise ValueError(f"metabolite display name mismatch: {feature}")
            names[feature] = values[0]
    return p, c, d, names


def sort_candidate_scores(scores):
    """Fixed lexicographic priority; no composite score or biological weighting."""
    return scores.sort_values(SORT_COLUMNS, ascending=SORT_ASCENDING, kind="mergesort").reset_index(drop=True)


def select_candidates(primary, comparison, direct_comparison, names, top_k, controls, seed):
    evidence = primary.merge(comparison[[*KEYS, "total_top20_support_count", "total_sign_agreement_fraction"]],
                             on=KEYS, validate="one_to_one")
    evidence["rank_percentile"] = evidence.total_outcome_rank / evidence.groupby(["modality", "outcome"]).feature.transform("size")
    scores = []
    for (modality, feature), group in evidence.groupby(["modality", "feature"], sort=True):
        high = group[group.total_outcome_rank <= 20]
        if high.empty:
            high = group[group.total_outcome_rank == group.total_outcome_rank.min()]
        scores.append({
            "modality": modality, "feature": feature,
            "feature_display": names[feature] if modality == "Metabolite" else feature,
            "primary_top20_outcome_count": int(group.total_outcome_rank.le(20).sum()),
            "primary_top50_outcome_count": int(group.total_outcome_rank.le(50).sum()),
            "primary_mean_total_rank": float(group.total_outcome_rank.mean()),
            "primary_best_total_rank": int(group.total_outcome_rank.min()),
            "high_rank_mean_top20_model_support": float(high.total_top20_support_count.mean()),
            "mean_total_sign_agreement": float(group.total_sign_agreement_fraction.mean()),
            "primary_direct_top20_outcome_count": int(group.direct_outcome_rank.le(20).sum()),
            "primary_mean_direct_rank": float(group.direct_outcome_rank.mean()),
            "mean_abs_direct_effect": float(group.direct_effect.abs().mean()),
            "mean_abs_total_effect": float(group.total_effect.abs().mean()),
            "mean_abs_indirect_effect": float(group.indirect_effect.abs().mean()),
            "direction_reversal_outcome_count": int((group.direct_effect * group.total_effect < 0).sum()),
            "best_total_rank_percentile": float(group.rank_percentile.min()),
            "mean_total_rank_percentile": float(group.rank_percentile.mean()),
        })
    scores = pd.DataFrame(scores)
    selected = []
    for index, modality in enumerate(MODALITIES):
        ordered = sort_candidate_scores(scores[scores.modality == modality])
        k, n = top_k[modality], controls[modality]
        if len(ordered) < k + n:
            raise ValueError(f"{modality}: {len(ordered)} features cannot supply Top{k} plus {n} controls")
        candidates = ordered.head(k).copy()
        candidates["candidate_role"] = "candidate"
        candidates["rank"] = np.arange(1, k + 1)
        candidates["selection_reason"] = "primary_total_top20_frequency;fixed_lexicographic_ties"
        remaining = ordered.iloc[k:].sort_values(
            ["best_total_rank_percentile", "mean_total_rank_percentile", "feature"],
            ascending=[False, False, True], kind="mergesort")
        pool = remaining.head(max(n, math.ceil(.2 * len(remaining)))).sort_values("feature")
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        control = pool.iloc[np.sort(rng.choice(len(pool), n, replace=False))].copy()
        control["candidate_role"] = "low_structural_effect_control"
        control["rank"] = np.arange(1, n + 1)
        control["selection_reason"] = "seeded_sample_lowest_effect_quintile;not_biological_negative"
        selected.extend([candidates, control])
    manifest = pd.concat(selected, ignore_index=True)[CANDIDATE_COLUMNS]
    # Preserve all four models' signed effects and ranks, and 4.2 support, per Outcome.
    model_cols = [col for col in comparison if any(col.endswith("_" + f) for f in EFFECT_FIELDS)]
    evidence_cols = [*KEYS, *model_cols, "total_top20_support_count", "total_sign_agreement_count", "total_sign_agreement_fraction"]
    detailed = manifest[["modality", "feature", "feature_display", "candidate_role"]].merge(
        comparison[evidence_cols], on=["modality", "feature"], validate="one_to_many")
    detailed = detailed.merge(direct_comparison[[*KEYS, "top20_support_count", "sign_agreement_fraction"]].rename(
        columns={"top20_support_count": "direct_top20_support_count", "sign_agreement_fraction": "direct_sign_agreement_fraction"}),
        on=KEYS, validate="one_to_one")
    detailed = detailed.merge(primary[[*KEYS, "direction_reversal", "effect_pattern"]].rename(
        columns={"direction_reversal": "primary_direction_reversal", "effect_pattern": "primary_effect_pattern"}),
        on=KEYS, validate="one_to_one")
    detailed = detailed.sort_values(KEYS, kind="mergesort").reset_index(drop=True)
    return manifest, detailed


def _eid_values(values, source):
    result = []
    for value in values:
        if isinstance(value, bool) or not str(value).lstrip("-").isdigit():
            raise ValueError(f"{source}: eid must be an integer, got {value!r}")
        result.append(int(value))
    if len(result) != len(set(result)):
        raise ValueError(f"{source}: duplicate patient eid")
    return result


def load_data(data_dir, config):
    tables = []
    for filename, cols in [
        ("anchors_structured.csv", config["anchor_cols"]),
        ("phenotype_structured.csv", config["phenotype_continuous_cols"] + config["phenotype_binary_cols"]),
        ("protein_full_structured.csv", config["protein_cols"]),
        ("metabolite_full_structured.csv", config["metabolite_cols"])]:
        path = data_dir / filename
        frame = _read_csv(path, ["eid"])
        missing = sorted(set(cols) - set(frame))
        if missing:
            raise ValueError(f"{path}: missing features: {missing}")
        frame["eid"] = _eid_values(frame.eid, str(path))
        frame = frame[["eid", *cols]].copy()
        frame[cols] = frame[cols].apply(pd.to_numeric, errors="raise")
        if np.isinf(frame[cols].to_numpy(dtype=float)).any():
            raise ValueError(f"{path}: infinite structured values")
        tables.append(frame)
    # Match load_structured_inputs: anchor INNER phenotype, then LEFT omics.
    merged = tables[0].merge(tables[1], on="eid", how="inner", validate="one_to_one")
    for frame in tables[2:]:
        merged = merged.merge(frame, on="eid", how="left", validate="one_to_one")
    if merged.empty:
        raise ValueError("no shared anchor/phenotype EIDs")
    merged = merged.sort_values("eid", kind="mergesort").set_index("eid")
    availability = {}
    molecular = []
    for modality, name in MODALITIES.items():
        values = merged[config[f"{name}_cols"]].to_numpy(dtype=np.float32)
        if np.isinf(values).any():
            raise ValueError(f"{modality}: values overflow model float32 inputs")
        # Same finite float32 mask/fraction/threshold as PartialMultiOmicsDataset;
        # no torch import, imputation, encoder, SCM or outcome decoder is required.
        availability[f"{name}_observed_fraction"] = np.isfinite(values).astype(np.float32).mean(axis=1)
        availability[f"{name}_available"] = availability[f"{name}_observed_fraction"] >= config.get(f"{name}_availability_threshold", .5)
        molecular.append(pd.DataFrame(values, columns=config[f"{name}_cols"], index=merged.index))
    availability["paired_available"] = availability["protein_available"] & availability["metabolite_available"]
    nonmolecular = config["anchor_cols"] + config["phenotype_continuous_cols"] + config["phenotype_binary_cols"]
    merged = pd.concat([merged[nonmolecular], *molecular, pd.DataFrame(availability, index=merged.index)], axis=1)
    params = _json(data_dir / "preprocessing_params.json")
    for name in MODALITIES.values():
        if not isinstance(params.get(name), dict):
            raise ValueError(f"preprocessing_params: missing/invalid {name} mapping")
        for feature in config[f"{name}_cols"]:
            entry = params.get(name, {}).get(feature, {})
            if not isinstance(entry, dict):
                raise ValueError(f"preprocessing_params: invalid {name}/{feature} entry")
            values = [entry.get("mean", np.nan), entry.get("std", np.nan)]
            if not np.isfinite(values).all() or float(values[1]) <= 0:
                raise ValueError(f"preprocessing_params: invalid/missing {name}/{feature} mean/std")
    return merged


def validate_split(split, data, summary, source):
    missing = [key for key in SPLITS if key not in split]
    if missing:
        raise ValueError(f"{source}: missing patient split lists: {missing}; cannot recover EIDs from counts")
    all_ids = set(data.index)
    result = {}
    for key in SPLITS:
        if not isinstance(split[key], list):
            raise ValueError(f"{source}: {key} must be a list")
        ids = set(_eid_values(split[key], f"{source}/{key}"))
        if ids - all_ids:
            raise ValueError(f"{source}: {key} has unknown EIDs: {sorted(ids-all_ids)[:20]}")
        count_key = "n_" + key.removesuffix("_eids")
        if count_key in split and split[count_key] != len(ids):
            raise ValueError(f"{source}: {count_key} mismatch")
        result[key.removesuffix("_eids")] = ids
    for stage in ("protein", "metabolite", "paired"):
        train, val = result[f"{stage}_train"], result[f"{stage}_val"]
        if train & val:
            raise ValueError(f"{source}: {stage} training/validation overlap")
        available = set(data.index[data[f"{stage}_available"]])
        if train | val != available:
            raise ValueError(f"{source}: {stage} split/availability mismatch")
        for metadata in (split, summary):
            count_key = f"n_{stage}_available" if metadata is split else f"{stage}_available_ge_threshold"
            if count_key in metadata and metadata[count_key] != len(available):
                raise ValueError(f"{source}: {count_key} mismatch")
    for metadata, key in [(split, "n_total"), (summary, "n_samples")]:
        if key in metadata and metadata[key] != len(data):
            raise ValueError(f"{source}: {key} != EID-joined patient count")
    return result


def build_cohort_and_plan(data, candidates, split, model, doses, quantiles):
    """Audit all EIDs; plan main and separately labeled single-modality validation."""
    flags = data[["protein_observed_fraction", "metabolite_observed_fraction", "protein_available",
                  "metabolite_available", "paired_available"]].copy()
    for stage, ids in split.items():
        flags[stage] = flags.index.isin(ids)
    training_union = split["protein_train"] | split["metabolite_train"] | split["paired_train"]
    flags["strictly_untrained"] = ~flags.index.isin(training_union)
    flags["not_joint_trained"] = ~flags.paired_train
    cohorts, plans, bounds = [], [], []
    for candidate in candidates.to_dict("records"):
        modality, feature = candidate["modality"], candidate["feature"]
        name = MODALITIES[modality]
        other = "metabolite" if name == "protein" else "protein"
        rows = flags.copy()
        rows["eid"] = rows.index
        for key in ("modality", "feature", "candidate_role"):
            rows[key] = candidate[key]
        rows["model"], rows["model_role"] = model["label"], model["role"]
        rows["baseline_value"] = data[feature]
        rows["candidate_observed"] = np.isfinite(rows.baseline_value)
        rows["candidate_missing"] = ~rows.candidate_observed
        main = rows.paired_val & rows.paired_available
        exploratory = rows[f"{name}_val"] & rows[f"{name}_available"] & ~rows[f"{other}_available"] & rows.not_joint_trained
        rows["cohort_type"] = np.select([main, exploratory], ["main_paired_validation", "exploratory_single_modality_validation"], default="excluded")
        rows["eligible_main"] = main & rows.candidate_observed
        rows["eligible_exploratory"] = exploratory & rows.candidate_observed
        rows["eligible"] = rows.eligible_main | rows.eligible_exploratory
        def reasons(row):
            why = []
            if row.cohort_type == "excluded":
                why.append("not_paired_validation_or_single_modality_validation")
            if not row[f"{name}_available"]:
                why.append("candidate_modality_unavailable")
            if not row.candidate_observed:
                why.append("candidate_missing")
            return ";".join(why) if why else "none"
        rows["exclusion_reason"] = rows.apply(reasons, axis=1)
        cohorts.append(rows.reset_index(drop=True)[COHORT_COLUMNS])
        # Feature actually used in modality pretraining or joint training; never
        # derive bounds from patients used exclusively for validation.
        training_ids = split[f"{name}_train"] | split["paired_train"]
        observed = data.loc[data.index.isin(training_ids), feature].dropna()
        lo, hi = (observed.quantile(quantiles, interpolation="linear").tolist() if len(observed) else [np.nan, np.nan])
        bounds.append({"model": model["label"], "modality": modality, "feature": feature,
                       "support_n": len(observed), "support_lower": float(lo) if len(observed) else None,
                       "support_upper": float(hi) if len(observed) else None})
        planned = rows.loc[main | exploratory].copy()
        for dose in doses:
            plan = planned.copy()
            plan["dose_sd"] = dose
            plan["target_value"] = plan.baseline_value + dose
            plan["is_baseline"] = dose == 0
            plan["support_lower"], plan["support_upper"], plan["support_n"] = lo, hi, len(observed)
            plan["out_of_support"] = (~plan.target_value.between(lo, hi)).astype("boolean")
            plan.loc[~plan.candidate_observed | (len(observed) == 0), "out_of_support"] = pd.NA
            why = []
            for row in plan.itertuples():
                blocked = []
                if not row.eligible:
                    blocked.append(row.exclusion_reason)
                if row.cohort_type != "main_paired_validation":
                    blocked.append("exploratory_requires_separate_executor")
                if not len(observed):
                    blocked.append("no_observed_training_support")
                elif row.candidate_observed and row.out_of_support and dose != 0:
                    blocked.append("out_of_support")
                why.append(";".join(blocked) if blocked else "none")
            plan["nonexecution_reason"] = why
            plan["executable"] = plan.nonexecution_reason.eq("none")
            plans.append(plan.reset_index(drop=True)[PLAN_COLUMNS])
    cohort = pd.concat(cohorts, ignore_index=True)
    plan = pd.concat(plans, ignore_index=True)
    return cohort, plan.sort_values(["model", "modality", "feature", "eid", "dose_sd"], kind="mergesort").reset_index(drop=True), bounds


def _model_report(cohort, plan, split, data, bounds):
    ids = set(data.index) - (split["protein_train"] | split["metabolite_train"] | split["paired_train"])
    counts = []
    for (modality, feature), rows in cohort.groupby(["modality", "feature"], sort=True):
        main = rows[rows.cohort_type == "main_paired_validation"]
        doses = plan[(plan.modality == modality) & (plan.feature == feature) & (plan.dose_sd != 0)
                     & (plan.cohort_type == "main_paired_validation")]
        counts.append({"modality": modality, "feature": feature,
            "eligible_main_n": int(rows.eligible_main.sum()), "eligible_exploratory_n": int(rows.eligible_exploratory.sum()),
            "strictly_untrained_main_n": int((rows.eligible_main & rows.strictly_untrained).sum()),
            "missing_rate_all": float(rows.candidate_missing.mean()),
            "missing_rate_paired_val": float(main.candidate_missing.mean()) if len(main) else None,
            "out_of_support_rate_nonzero_observed_main": float(doses.out_of_support.dropna().mean()) if doses.out_of_support.notna().any() else None,
            "executable_nonzero_rows": int(doses.executable.sum())})
    return {"status": "validated", "all_patient_count": len(data), "paired_val_count": len(split["paired_val"]),
            "strictly_untrained_count": len(ids),
            "strictly_untrained_available_any_count": int(data.loc[data.index.isin(ids), ["protein_available", "metabolite_available"]].any(axis=1).sum()),
            "strictly_untrained_paired_count": int(data.loc[data.index.isin(ids), "paired_available"].sum()),
            "eligible_main_unique_patients": int(cohort.loc[cohort.eligible_main, "eid"].nunique()),
            "strictly_untrained_main_unique_patients": int(cohort.loc[cohort.eligible_main & cohort.strictly_untrained, "eid"].nunique()),
            "per_candidate": counts, "training_support": bounds}


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-dir", type=Path, default=ROOT / "artifacts/analysis/direct_effect_screening")
    parser.add_argument("--total-dir", type=Path, default=total.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/structured_full")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/aclf_full_multimodal_config.json")
    parser.add_argument("--primary-dir", type=Path, default=total.DEFAULT_PRIMARY_DIR)
    parser.add_argument("--replication-dirs", type=Path, nargs=2, default=list(total.DEFAULT_REPLICATION_DIRS))
    parser.add_argument("--sensitivity-dirs", type=Path, nargs=1, default=list(total.DEFAULT_SENSITIVITY_DIRS))
    parser.add_argument("--protein-top-k", type=int, default=10)
    parser.add_argument("--metabolite-top-k", type=int, default=10)
    parser.add_argument("--controls-per-modality", type=int, default=3)
    parser.add_argument("--protein-controls", type=int, default=None)
    parser.add_argument("--metabolite-controls", type=int, default=None)
    parser.add_argument("--doses", type=float, nargs="+", default=[-1., -.5, 0., .5, 1.])
    parser.add_argument("--support-quantiles", type=float, nargs=2, default=[.01, .99], metavar=("LOWER", "UPPER"))
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/analysis/virtual_perturbation/phase1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and write only preflight_report.json; no formal manifests")
    return parser


def _validate_paths_and_options(args, specs):
    output = args.out_dir.resolve()
    protected = [args.direct_dir, args.total_dir, args.data_dir, args.config, *[s["directory"] for s in specs], ROOT / "src", ROOT / "scripts", ROOT / "tests", ROOT / "configs"]
    for path in protected:
        path = Path(path).resolve()
        if output == path or output in path.parents or path in output.parents:
            raise ValueError(f"output is in/contains a protected input/code path: {path}")
    if min(args.protein_top_k, args.metabolite_top_k) < 1 or min(args.controls_per_modality,
            args.protein_controls if args.protein_controls is not None else args.controls_per_modality,
            args.metabolite_controls if args.metabolite_controls is not None else args.controls_per_modality) < 0:
        raise ValueError("Top-K must be positive and controls nonnegative")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    if not np.isfinite(args.doses).all() or len(args.doses) != len(set(args.doses)) or 0. not in args.doses:
        raise ValueError("doses must be finite, unique and contain 0 SD baseline")
    if not 0 <= args.support_quantiles[0] < args.support_quantiles[1] <= 1:
        raise ValueError("support quantiles must satisfy 0 <= lower < upper <= 1")


def run_preparation(args):
    specs = total._model_specs(args.primary_dir, args.replication_dirs, args.sensitivity_dirs)
    _validate_paths_and_options(args, specs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": "4.4.1.v1", "status": "checking", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "dry_run": args.dry_run, "errors": [], "warnings": [], "missing_files": [], "models": {}}
    code_paths = [Path(__file__), Path(total.__file__), Path(direct.__file__), ROOT / "src/casualvae/module_aclf_partial_multimodal.py"]
    report["code_version"] = {"git_commit": _git("rev-parse", "HEAD"), "git_worktree_status": _git("status", "--porcelain"),
        "code_files": [{"path": str(p.resolve()), "sha256": _hash(p)} for p in code_paths]}
    # Invalidate any previous readiness before touching formal outputs. Consumers
    # must gate on this report and verify the final artifact hashes.
    _write_json(args.out_dir / "preflight_report.json", report)
    paths = [args.config, *[args.direct_dir / f for f in ("outcome_specific_cross_model_comparison.csv", "multi_outcome_feature_summary.csv", "direct_effect_summary.json")],
             *[args.total_dir / f for f in ("total_effect_primary.csv", "total_effect_cross_model_comparison.csv", "multi_outcome_total_effect_summary.csv", "total_effect_summary.json")],
             *[args.data_dir / f for f in ("anchors_structured.csv", "phenotype_structured.csv", "protein_full_structured.csv", "metabolite_full_structured.csv", "preprocessing_params.json")],
             *[Path(s["directory"]) / f for s in specs for f in ("model.pt", "used_config.json", "run_summary.json", "data_split_summary.json")]]
    def fail():
        report["status"] = "blocked"
        _write_json(args.out_dir / "preflight_report.json", report)
        raise PreflightError("Preflight blocked:\n" + "\n".join(report["errors"]))
    report["missing_files"] = [str(p.resolve()) for p in paths if not p.is_file()]
    if report["missing_files"]:
        report["errors"] = ["missing file: " + p for p in report["missing_files"]]
        fail()
    report["input_files"] = [{"path": str(p.resolve()), "sha256": _hash(p)} for p in paths]
    try:
        config = _json(args.config)
        _validate_config(config)
        for directory, file in [(args.direct_dir, "direct_effect_summary.json"), (args.total_dir, "total_effect_summary.json")]:
            summary = _json(directory / file)
            registered = summary["models"]
            items = [registered["primary"], *registered["replication"], *registered["sensitivity"]]
            if {i["label"] for i in items} != {s["label"] for s in specs} or len(items) != len(specs):
                raise ValueError(f"{file}: model labels mismatch")
            for spec in specs:
                item = next(i for i in items if i["label"] == spec["label"])
                if Path(item["directory"]).resolve() != Path(spec["directory"]).resolve() or item["role"] != spec["role"]:
                    raise ValueError(f"{file}: model directory/role mismatch for {spec['label']}")
            if set(summary["outcomes"]) != set(config["phenotype_continuous_cols"] + config["phenotype_binary_cols"]):
                raise ValueError(f"{file}: Outcome/config mismatch")
        primary, comparison, dc, names = load_effects(args, config, specs)
        top_k = {"Protein": args.protein_top_k, "Metabolite": args.metabolite_top_k}
        controls = {m: getattr(args, n + "_controls") if getattr(args, n + "_controls") is not None else args.controls_per_modality for m, n in MODALITIES.items()}
        candidates, evidence = select_candidates(primary, comparison, dc, names, top_k, controls, args.seed)
        report["selection_preview"] = candidates.to_dict("records")
        report["candidate_counts"] = candidates.groupby(["modality", "candidate_role"]).size().rename("count").reset_index().to_dict("records")
        data = load_data(args.data_dir, config)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        report["errors"].append(str(exc)); fail()
    cohorts, plans, registry = [], [], []
    for spec in specs:
        label, directory = spec["label"], Path(spec["directory"])
        try:
            used, run, split_json = [_json(directory / f) for f in ("used_config.json", "run_summary.json", "data_split_summary.json")]
            _validate_config(used)
            legacy = [k for k in ("outcome_hidden_dims",) if k in used]
            if "outcome" in used.get("hidden_dims", {}):
                legacy.append("hidden_dims.outcome")
            if legacy:
                report["warnings"].append({"code": "inactive_legacy_config_fields", "model": label, "fields": legacy,
                    "message": "Unused by current model constructor. Raw metadata preserved; checkpoint architecture still requires strict phase 2 validation."})
            for key in CONFIG_KEYS:
                if _active_config_value(used, key) != _active_config_value(config, key):
                    raise ValueError(f"{label}: model config mismatch in {key}")
            if (directory / "model.pt").stat().st_size == 0:
                raise ValueError(f"{label}: empty model.pt")
            metadata_errors = []
            for key in ("split_seed", "model_seed"):
                if type(run.get(key)) is not int or run.get(key) != split_json.get(key):
                    metadata_errors.append(f"missing/inconsistent {key}; directory names are not seed provenance")
            try:
                split = validate_split(split_json, data, run, label)
            except (ValueError, TypeError) as exc:
                metadata_errors.append(str(exc))
            if metadata_errors:
                raise ValueError(f"{label}: " + "; ".join(metadata_errors))
            for key in ("shared_dims", "private_dims", "outcome_exogenous_mode", "outcome_exogenous_sigma"):
                if key in run and run[key] != _active_config_value(used, key):
                    raise ValueError(f"{label}: run_summary/config mismatch in {key}")
            for key, count in [("anchor_features", len(used["anchor_cols"])),
                ("protein_features", len(used["protein_cols"])), ("metabolite_features", len(used["metabolite_cols"])),
                ("phenotype_cont_features", 14), ("phenotype_bin_features", 3)]:
                if key in run and run[key] != count:
                    raise ValueError(f"{label}: run_summary/config mismatch in {key}")
            tau = run.get("best_tau")
            if not isinstance(tau, (float, int)) or not np.isfinite(tau) or tau <= 0:
                raise ValueError(f"{label}: missing/invalid best_tau")
            model = {"label": label, "role": spec["role"], "directory": str(directory.resolve()),
                     "weights_path": str((directory / "model.pt").resolve()), "weights_sha256": _hash(directory / "model.pt"),
                     "split_seed": run["split_seed"], "model_seed": run["model_seed"], "best_tau": tau, "used_config": used}
            cohort, plan, bounds = build_cohort_and_plan(data, candidates, split, model, sorted(args.doses), args.support_quantiles)
            report["models"][label] = _model_report(cohort, plan, split, data, bounds)
            model_report = report["models"][label]
            if model_report["strictly_untrained_main_unique_patients"] < 10:
                report["warnings"].append({"code": "few_strictly_untrained_main", "model": label, "count": model_report["strictly_untrained_main_unique_patients"]})
            if not model_report["eligible_main_unique_patients"]:
                report["warnings"].append({"code": "empty_main_cohort", "model": label})
            for bound in bounds:
                if bound["support_n"] < 2:
                    report["warnings"].append({"code": "insufficient_training_support", **bound})
            cohorts.append(cohort); plans.append(plan); registry.append(model)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            report["errors"].append(str(exc))
            report["models"][label] = {"status": "blocked", "error": str(exc)}
    report["warnings"].extend([
        {"code": "legacy_global_standardization", "message": "Existing preprocessing computes SD over available full structured source data before model splits. Phase 1 preserves model input units; only support quantiles use model training EIDs. This does not establish leakage-free external validation."},
        {"code": "phase2_encoder_sampling", "message": PHASE2_REQUIREMENTS["encoder_sampling"]}])
    if report["errors"]:
        fail()
    if args.dry_run:
        report["status"] = "validated"
        report.pop("selection_preview", None)
        _write_json(args.out_dir / "preflight_report.json", report)
        return report
    cohort = pd.concat(cohorts, ignore_index=True).sort_values(["model", "modality", "feature", "eid"], kind="mergesort").reset_index(drop=True)
    plan = pd.concat(plans, ignore_index=True).sort_values(["model", "modality", "feature", "eid", "dose_sd"], kind="mergesort").reset_index(drop=True)
    outputs = {"candidate_manifest.csv": candidates, "candidate_outcome_evidence.csv": evidence,
               "cohort_manifest.csv": cohort, "perturbation_plan.csv": plan}
    output_info = {}
    for filename, frame in outputs.items():
        path = args.out_dir / filename
        temporary = path.with_suffix(".csv.tmp")
        frame.to_csv(temporary, index=False, float_format="%.17g", lineterminator="\n")
        temporary.replace(path)
        output_info[filename] = {"path": str(path.resolve()), "sha256": _hash(path), "rows": len(frame), "columns": frame.columns.tolist()}
    continuous, binary = config["phenotype_continuous_cols"], config["phenotype_binary_cols"]
    liver = [c for c in continuous if c in ("ALT", "AST", "ALP", "GGT", "TBil", "DBil", "ALB")]
    experiment = {
        "schema_version": report["schema_version"], "phase": "4.4.1", "generated_at_utc": report["generated_at_utc"],
        **report["code_version"],
        "input_files": report["input_files"],
        "input_directories": {k: str(getattr(args, k).resolve()) for k in ("direct_dir", "total_dir", "data_dir")},
        "models": registry, "outcomes": {"continuous": continuous, "binary": binary, "ordered": continuous + binary,
            "report_groups": {"liver_continuous": liver, "other_continuous": [c for c in continuous if c not in liver], "hepatic_failure_binary": binary},
            "label_interpretation": "Recorded hepatic failure labels; not confirmed EASL-CLIF ACLF diagnoses."},
        "selection": {"top_k": top_k, "controls": controls, "sort_columns": SORT_COLUMNS, "sort_ascending": SORT_ASCENDING,
            "mean_rank_scope": "all 17 Outcomes, separately within modality", "support_denominator": len(specs), "sign_agreement_denominator": len(specs)-1,
            "high_rank_edges": "Primary total rank <=20; if empty use edges at the feature's best Primary total rank",
            "control_rule": "Exclude candidates; order remaining by best Total rank percentile descending, mean Total rank percentile descending, feature ascending. Pool size=max(control_count,ceil(0.2*remaining_count)); sample sorted feature codes without replacement using SeedSequence([seed,modality_index]); modality order Protein,Metabolite.",
            "no_biological_weights": True},
        "cohort_rules": {"join": "anchors INNER phenotype LEFT protein LEFT metabolite ON eid, one-to-one; sort eid",
            "availability": "finite float32 fraction over configured feature order >= model threshold; candidate must be finite",
            "main": "paired_val_eids AND paired_available AND candidate_observed",
            "strictly_untrained": "all joined EIDs minus union(protein_train_eids,metabolite_train_eids,paired_train_eids)",
            "exploratory": "own modality validation and available, other modality unavailable, not paired training; recorded separately, joint executor disabled"},
        "dose_design": {"doses_sd": sorted(args.doses), "unit": "existing standardized input SD", "operation": "target=baseline+dose_sd; no clipping",
            "support_quantiles": args.support_quantiles, "quantile_method": "linear",
            "support_training_eids": "union(own_modality_train_eids,paired_train_eids), observed feature values only",
            "zero_sd_policy": "retain baseline and support flag; eligible observed zero dose executable even outside quantiles; no-support baselines disabled",
            "extrapolation": "nonzero out-of-support rows disabled"},
        "random_seed": args.seed, "outputs": output_info, "phase2_requirements": PHASE2_REQUIREMENTS,
        "interpretation": "Input-feature perturbation plans only, not strict SCM do interventions; no predictions or clinical target claims."}
    _write_json(args.out_dir / "experiment_config.json", experiment)
    report["status"] = "ready"
    report["outputs"] = output_info
    report["experiment_config_sha256"] = _hash(args.out_dir / "experiment_config.json")
    report.pop("selection_preview", None)
    _write_json(args.out_dir / "preflight_report.json", report)
    return report


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    try:
        report = run_preparation(args)
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "out_dir": str(args.out_dir.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
