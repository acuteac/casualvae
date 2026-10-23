"""Phase 1 tests use only synthetic patients, effects and model-file bytes."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analysis import direct_effect_screening as direct
from scripts.analysis import total_effect_propagation as total

try:
    from scripts.analysis import prepare_virtual_perturbation as phase1
except ImportError:
    phase1 = None

CONT = ["CRP", "WBC", "NEUT", "NEUT_P", "PLT", "ALT", "AST", "ALP",
        "GGT", "TBil", "DBil", "ALB", "Urea", "Crea"]
BIN = ["acute_or_subacute_hepatic_failure", "chronic_hepatic_failure",
       "hepatic_failure_unspecified"]
LABELS = ["primary", "variable456", "variable789", "fixed789"]


def fixture(root):
    dirs = {name: root / name for name in ["direct", "total", "data", *LABELS]}
    for path in dirs.values():
        path.mkdir()
    config = {
        "anchor_cols": ["sex"], "phenotype_continuous_cols": CONT,
        "phenotype_binary_cols": BIN,
        "protein_cols": [f"P{i:03}" for i in range(65)],
        "metabolite_cols": [f"m{i:03}" for i in range(65)],
        "protein_availability_threshold": .5, "metabolite_availability_threshold": .5,
        "shared_dims": {"protein": 2, "metabolite": 2, "outcome": 1},
        "private_dims": {"protein": 1, "metabolite": 1},
        "hidden_dims": {"protein": [4], "metabolite": [4]},
        "outcome_exogenous_mode": "zero", "outcome_exogenous_sigma": 1.,
    }
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config))
    models = {}
    for label in LABELS:
        blocks = []
        for modality, key in [("Protein", "protein_cols"), ("Metabolite", "metabolite_cols")]:
            values = np.tile(np.arange(65, 0, -1, dtype=float), (17, 1))
            matrix = pd.DataFrame(values, index=CONT + BIN, columns=config[key])
            blocks.append(total.build_model_edge_table(matrix * .9, matrix, modality))
        models[label] = pd.concat(blocks, ignore_index=True)
    roles = dict(zip(LABELS, ["primary", "replication", "replication", "sensitivity"]))
    comparison = total.build_cross_model_comparison(models, roles)
    comparison["metabolite_name"] = comparison["feature"].map(lambda f: "Display " + f)
    primary = models["primary"].copy()
    primary["metabolite_name"] = primary["feature"].map(lambda f: "Display " + f)
    primary.to_csv(dirs["total"] / "total_effect_primary.csv", index=False)
    comparison.to_csv(dirs["total"] / "total_effect_cross_model_comparison.csv", index=False)
    total.build_multi_outcome_summary(primary).to_csv(
        dirs["total"] / "multi_outcome_total_effect_summary.csv", index=False)
    direct_models = {label: frame.rename(columns={"direct_effect": "effect",
        "direct_outcome_rank": "outcome_rank"}).assign(
            abs_effect=frame.direct_effect.abs()) for label, frame in models.items()}
    dc = direct.build_cross_model_comparison(direct_models, roles)
    dc["metabolite_name"] = dc["feature"].map(lambda f: "Display " + f)
    dc.to_csv(dirs["direct"] / "outcome_specific_cross_model_comparison.csv", index=False)
    direct.build_multi_outcome_summary(direct_models["primary"]).to_csv(
        dirs["direct"] / "multi_outcome_feature_summary.csv", index=False)
    registry = {"primary": {"label": "primary", "role": "primary", "directory": str(dirs["primary"])},
                "replication": [{"label": l, "role": "replication", "directory": str(dirs[l])}
                                for l in LABELS[1:3]],
                "sensitivity": [{"label": "fixed789", "role": "sensitivity", "directory": str(dirs["fixed789"])}]}
    for name, file in [("direct", "direct_effect_summary.json"), ("total", "total_effect_summary.json")]:
        (dirs[name] / file).write_text(json.dumps({"models": registry, "outcomes": CONT + BIN}))
    split = {"protein_train_eids": [1, 2, 3], "protein_val_eids": [4, 5, 6],
             "metabolite_train_eids": [1, 2, 4], "metabolite_val_eids": [3, 5],
             "paired_train_eids": [1, 2, 3], "paired_val_eids": [4, 5],
             "n_total": 6, "split_seed": 42, "model_seed": 456}
    for label in LABELS:
        path = dirs[label]
        (path / "model.pt").write_bytes(b"synthetic fixture: never deserialized")
        (path / "used_config.json").write_text(json.dumps(config))
        (path / "run_summary.json").write_text(json.dumps({"split_seed": 42, "model_seed": 456,
            "best_tau": .9, "n_samples": 6}))
        (path / "data_split_summary.json").write_text(json.dumps(split))
    ids = [6, 2, 5, 1, 4, 3]
    pd.DataFrame({"eid": ids, "sex": [0]*6}).to_csv(dirs["data"] / "anchors_structured.csv", index=False)
    pd.DataFrame({"eid": ids, **{c: [0.]*6 for c in CONT + BIN}}).to_csv(
        dirs["data"] / "phenotype_structured.csv", index=False)
    params = {}
    for name, file in [("protein", "protein_full_structured.csv"), ("metabolite", "metabolite_full_structured.csv")]:
        frame = pd.DataFrame({"eid": ids, **{c: [float(i-1) for i in ids] for c in config[name+"_cols"]}})
        if name == "metabolite":
            frame.loc[frame.eid == 6, config["metabolite_cols"]] = np.nan
        frame.sample(frac=1, random_state=7).to_csv(dirs["data"] / file, index=False)
        params[name] = {c: {"mean": 0., "std": 1.} for c in config[name+"_cols"]}
    (dirs["data"] / "preprocessing_params.json").write_text(json.dumps(params))
    args = ["--direct-dir", str(dirs["direct"]), "--total-dir", str(dirs["total"]),
            "--data-dir", str(dirs["data"]), "--config", str(config_path),
            "--primary-dir", str(dirs["primary"]), "--replication-dirs",
            str(dirs["variable456"]), str(dirs["variable789"]),
            "--sensitivity-dirs", str(dirs["fixed789"]), "--protein-top-k", "2",
            "--metabolite-top-k", "2", "--controls-per-modality", "2",
            "--out-dir", str(root / "out")]
    return dirs, args


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(phase1, "phase1 implementation must exist")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dirs, argv = fixture(self.root)
        self.args = phase1.build_argument_parser().parse_args(argv)

    def run_ok(self):
        report = phase1.run_preparation(self.args)
        self.assertEqual(report["status"], "ready")
        return report

    def csv(self, file):
        return pd.read_csv(self.args.out_dir / file)

    def test_top_k_names_controls_and_repeatability(self):
        self.run_ok()
        manifest = self.csv("candidate_manifest.csv")
        for modality, prefix in [("Protein", "P"), ("Metabolite", "m")]:
            rows = manifest[manifest.modality == modality]
            selected = rows[rows.candidate_role == "candidate"]
            self.assertEqual(selected.feature.tolist(), [prefix+"000", prefix+"001"])
            self.assertFalse(set(selected.feature) & set(rows[rows.candidate_role == "low_structural_effect_control"].feature))
        self.assertEqual(manifest.set_index("feature").loc["m000", "feature_display"], "Display m000")
        before = {f: (self.args.out_dir / f).read_bytes() for f in ["candidate_manifest.csv", "cohort_manifest.csv", "perturbation_plan.csv"]}
        self.run_ok()
        for f, data in before.items():
            self.assertEqual(data, (self.args.out_dir / f).read_bytes())

    def test_all_tie_break_directions(self):
        base = dict(primary_top20_outcome_count=3, primary_top50_outcome_count=5,
                    primary_mean_total_rank=30., high_rank_mean_top20_model_support=2.,
                    mean_total_sign_agreement=0.5)
        directions = [("primary_top20_outcome_count", 4), ("primary_top50_outcome_count", 6),
                      ("primary_mean_total_rank", 29.), ("high_rank_mean_top20_model_support", 3.),
                      ("mean_total_sign_agreement", .8)]
        for field, better in directions:
            rows = pd.DataFrame([{**base, "feature": "a"}, {**base, field: better, "feature": "z"}])
            self.assertEqual(phase1.sort_candidate_scores(rows).feature.tolist(), ["z", "a"], field)
        rows = pd.DataFrame([{**base, "feature": "z"}, {**base, "feature": "a"}])
        self.assertEqual(phase1.sort_candidate_scores(rows).feature.tolist(), ["a", "z"])
        # Earlier criteria dominate arbitrarily favorable later criteria.
        for i, (field, better) in enumerate(directions):
            worse_later = dict(base)
            for later, _ in directions[i+1:]:
                worse_later[later] = 1e6 if later == "primary_mean_total_rank" else -1e6
            rows = pd.DataFrame([{**base, "feature": "a"}, {**worse_later, field: better, "feature": "z"}])
            self.assertEqual(phase1.sort_candidate_scores(rows).feature.tolist(), ["z", "a"])

    def test_primary_frequency_is_not_model_support(self):
        self.run_ok()
        row = self.csv("candidate_manifest.csv").iloc[0]
        self.assertEqual(row.primary_top20_outcome_count, 17)
        self.assertEqual(row.high_rank_mean_top20_model_support, 4.)

    def test_recurrence_dominates_mean_rank_and_cross_model_support(self):
        primary = pd.read_csv(self.dirs["total"] / "total_effect_primary.csv")
        comparison = pd.read_csv(self.dirs["total"] / "total_effect_cross_model_comparison.csv")
        dc = pd.read_csv(self.dirs["direct"] / "outcome_specific_cross_model_comparison.csv")
        # P000 is extremely high for 16 outcomes, but misses Top20 once.
        # P019 is only rank 20 yet recurs in all 17 and must outrank P000.
        primary.loc[(primary.feature == "P000") & (primary.outcome == "ALT"), "total_outcome_rank"] = 21
        comparison.loc[comparison.feature == "P019", "total_top20_support_count"] = 1
        names = {f"m{i:03}": "Display m" + f"{i:03}" for i in range(65)}
        manifest, _ = phase1.select_candidates(primary, comparison, dc, names,
            {"Protein": 20, "Metabolite": 2}, {"Protein": 1, "Metabolite": 1}, 42)
        selected = manifest[(manifest.modality == "Protein") & (manifest.candidate_role == "candidate")]
        self.assertLess(selected.feature.tolist().index("P019"), selected.feature.tolist().index("P000"))

    def test_eid_join_missingness_and_distinct_validation_flags(self):
        path = self.dirs["data"] / "protein_full_structured.csv"
        frame = pd.read_csv(path)
        frame.loc[frame.eid == 5, "P000"] = np.nan
        frame.to_csv(path, index=False)
        self.run_ok()
        cohort = self.csv("cohort_manifest.csv")
        rows = cohort[(cohort.model == "primary") & (cohort.feature == "P000")].set_index("eid")
        self.assertTrue(rows.loc[4, "paired_val"])
        self.assertFalse(rows.loc[4, "strictly_untrained"])
        self.assertTrue(rows.loc[5, "strictly_untrained"])
        self.assertFalse(rows.loc[5, "candidate_observed"])
        self.assertIn("candidate_missing", rows.loc[5, "exclusion_reason"])
        self.assertEqual(rows.loc[4, "baseline_value"], 3.)
        self.assertEqual(rows.loc[6, "cohort_type"], "exploratory_single_modality_validation")
        plan = self.csv("perturbation_plan.csv")
        self.assertFalse(plan[(plan.eid == 5) & (plan.feature == "P000")].executable.any())
        before = (self.args.out_dir / "cohort_manifest.csv").read_bytes()
        for path in self.dirs["data"].glob("*.csv"):
            pd.read_csv(path).sample(frac=1, random_state=13).to_csv(path, index=False)
        self.run_ok()
        self.assertEqual(before, (self.args.out_dir / "cohort_manifest.csv").read_bytes())

    def test_doses_training_only_bounds_and_zero_baseline(self):
        self.run_ok()
        plan = self.csv("perturbation_plan.csv")
        rows = plan[(plan.model == "primary") & (plan.feature == "P000") & (plan.eid == 4)].set_index("dose_sd")
        self.assertEqual(sorted(rows.index), [-1., -.5, 0., .5, 1.])
        self.assertAlmostEqual(rows.loc[1., "target_value"], 4.)
        self.assertAlmostEqual(rows.loc[0., "support_lower"], .02)
        self.assertAlmostEqual(rows.loc[0., "support_upper"], 1.98)
        self.assertEqual(rows.loc[0., "support_n"], 3)
        self.assertTrue(rows.loc[0., "out_of_support"])
        self.assertTrue(rows.loc[0., "executable"])
        self.assertFalse(rows.loc[1., "executable"])
        self.assertEqual(rows.loc[1., "nonexecution_reason"], "out_of_support")
        exploratory = plan[plan.cohort_type == "exploratory_single_modality_validation"]
        self.assertFalse(exploratory.executable.any())

    def test_config_order_hashes_and_no_prediction(self):
        self.run_ok()
        config = json.loads((self.args.out_dir / "experiment_config.json").read_text())
        self.assertEqual(config["outcomes"]["continuous"], CONT)
        self.assertEqual(config["outcomes"]["binary"], BIN)
        self.assertEqual(config["models"][0]["best_tau"], .9)
        self.assertEqual(len(config["input_files"][0]["sha256"]), 64)
        self.assertIn("encoder_sampling", config["phase2_requirements"])

    def test_dry_run_has_only_report(self):
        self.args.dry_run = True
        report = phase1.run_preparation(self.args)
        self.assertEqual(report["status"], "validated")
        self.assertEqual([p.name for p in self.args.out_dir.iterdir()], ["preflight_report.json"])

    def test_inactive_legacy_metadata_warns_but_active_architecture_mismatch_fails(self):
        for label in LABELS:
            path = self.dirs[label] / "used_config.json"
            config = json.loads(path.read_text())
            config["hidden_dims"]["outcome"] = [64, 32]
            config["outcome_hidden_dims"] = [64, 32]
            path.write_text(json.dumps(config))
        report = self.run_ok()
        self.assertTrue(any(w["code"] == "inactive_legacy_config_fields" for w in report["warnings"]))
        path = self.dirs["primary"] / "used_config.json"
        config = json.loads(path.read_text()); config["hidden_dims"]["protein"] = [999]
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(phase1.PreflightError, "hidden_dims"):
            phase1.run_preparation(self.args)

    def test_no_training_observations_disables_all_doses_and_warns(self):
        path = self.dirs["data"] / "protein_full_structured.csv"
        frame = pd.read_csv(path)
        frame.loc[frame.eid.isin([1, 2, 3]), "P000"] = np.nan
        frame.to_csv(path, index=False)
        report = self.run_ok()
        rows = self.csv("perturbation_plan.csv")
        rows = rows[(rows.model == "primary") & (rows.feature == "P000")]
        self.assertFalse(rows.executable.any())
        self.assertTrue(rows.out_of_support.isna().all())
        self.assertTrue(any(w["code"] == "insufficient_training_support" for w in report["warnings"]))

    def test_empty_validation_is_reported_and_zero_controls_supported(self):
        self.args.controls_per_modality = 0
        for label in LABELS:
            path = self.dirs[label] / "data_split_summary.json"
            split = json.loads(path.read_text())
            split["paired_train_eids"] += split["paired_val_eids"]
            split["paired_val_eids"] = []
            path.write_text(json.dumps(split))
        report = self.run_ok()
        self.assertTrue(self.csv("candidate_manifest.csv").candidate_role.eq("candidate").all())
        self.assertEqual(report["models"]["primary"]["eligible_main_unique_patients"], 0)

    def test_nonempty_dry_run_does_not_authorize_previous_manifests(self):
        self.run_ok()
        self.args.dry_run = True
        report = phase1.run_preparation(self.args)
        self.assertEqual(report["status"], "validated")
        self.assertTrue((self.args.out_dir / "candidate_manifest.csv").exists())
        self.assertNotIn("outputs", report)

    def test_missing_files_all_listed(self):
        (self.dirs["primary"] / "model.pt").unlink()
        (self.dirs["data"] / "anchors_structured.csv").unlink()
        with self.assertRaisesRegex(phase1.PreflightError, "model.pt"):
            phase1.run_preparation(self.args)
        report = json.loads((self.args.out_dir / "preflight_report.json").read_text())
        self.assertEqual(len(report["missing_files"]), 2)
        self.assertFalse((self.args.out_dir / "candidate_manifest.csv").exists())

    def test_missing_historical_split_ids_is_blocker_not_reconstructed(self):
        path = self.dirs["variable456"] / "data_split_summary.json"
        path.write_text(json.dumps({"n_total": 6}))
        with self.assertRaisesRegex(phase1.PreflightError, "protein_train_eids"):
            phase1.run_preparation(self.args)
        report = json.loads((self.args.out_dir / "preflight_report.json").read_text())
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["models"]["variable456"]["status"], "blocked")
        self.assertEqual(report["models"]["primary"]["paired_val_count"], 2)

    def test_anomalous_inputs_fail_clearly(self):
        cases = ["duplicate_eid", "nonfinite_effect", "wrong_modality", "duplicate_feature",
                 "split_overlap", "unknown_split_eid", "missing_feature", "config_mismatch", "rank_mismatch"]
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                dirs, argv = fixture(Path(tmp))
                args = phase1.build_argument_parser().parse_args(argv)
                if case == "duplicate_eid":
                    path = dirs["data"] / "anchors_structured.csv"
                    frame = pd.read_csv(path); pd.concat([frame, frame.iloc[:1]]).to_csv(path, index=False)
                elif case in ["nonfinite_effect", "wrong_modality", "rank_mismatch"]:
                    path = dirs["total"] / "total_effect_primary.csv"
                    frame = pd.read_csv(path)
                    frame.loc[0, {"nonfinite_effect": "total_effect", "wrong_modality": "modality", "rank_mismatch": "total_outcome_rank"}[case]] = {
                        "nonfinite_effect": np.inf, "wrong_modality": "Metabolite", "rank_mismatch": 2}[case]
                    frame.to_csv(path, index=False)
                elif case in ["duplicate_feature", "config_mismatch"]:
                    path = dirs["primary"] / "used_config.json"
                    data = json.loads(path.read_text()); data["protein_cols"] = data["protein_cols"] + ["P000"] if case == "duplicate_feature" else list(reversed(data["protein_cols"]))
                    path.write_text(json.dumps(data))
                elif case in ["split_overlap", "unknown_split_eid"]:
                    path = dirs["primary"] / "data_split_summary.json"
                    data = json.loads(path.read_text()); data["paired_val_eids"].append(1 if case == "split_overlap" else 999)
                    path.write_text(json.dumps(data))
                else:
                    path = dirs["data"] / "protein_full_structured.csv"
                    pd.read_csv(path).drop(columns="P000").to_csv(path, index=False)
                with self.assertRaises(phase1.PreflightError):
                    phase1.run_preparation(args)

    def test_protected_output_and_invalid_doses(self):
        self.args.out_dir = self.dirs["total"]
        with self.assertRaisesRegex(ValueError, "protected"):
            phase1.run_preparation(self.args)
        self.args.out_dir = self.root / "out"
        self.args.doses = [-1., 1.]
        with self.assertRaisesRegex(ValueError, "0 SD"):
            phase1.run_preparation(self.args)

    def test_invalid_architecture_types_and_noise_never_publish_readiness(self):
        for field, value in [("shared_dims", None),
                             ("shared_dims", {"protein": -1, "metabolite": 2, "outcome": 1}),
                             ("private_dims", {"protein": -1, "metabolite": 1}),
                             ("hidden_dims", {"protein": [0], "metabolite": [4]}),
                             ("outcome_exogenous_sigma", -1.), ("outcome_exogenous_sigma", float("nan"))]:
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as tmp:
                dirs, argv = fixture(Path(tmp))
                args = phase1.build_argument_parser().parse_args(argv)
                for path in [args.config, *[dirs[l] / "used_config.json" for l in LABELS]]:
                    config = json.loads(path.read_text()); config[field] = value
                    path.write_text(json.dumps(config))
                with self.assertRaises(phase1.PreflightError):
                    phase1.run_preparation(args)
                report = json.loads((args.out_dir / "preflight_report.json").read_text())
                self.assertEqual(report["status"], "blocked")

    def test_help(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/analysis/prepare_virtual_perturbation.py"), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--dry-run", result.stdout)


if __name__ == "__main__":
    unittest.main()
