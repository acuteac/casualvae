import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analysis import analyze_multiseed


class TrainingSummaryTests(unittest.TestCase):
    def test_extracts_real_joint_history_columns_at_best_epoch_without_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "run_summary.json").write_text(
                json.dumps({"best_epoch": 2, "best_tau": 1.25, "split_seed": 42, "model_seed": 123}),
                encoding="utf-8",
            )
            pd.DataFrame(
                {
                    "stage": ["pretrain_protein", "joint", "joint"],
                    "epoch": [2, 1, 2],
                    "val_loss": [9.0, 1.2, 0.8],
                    "val_outcome_cont": [np.nan, 0.5, 0.4],
                    "val_outcome_bin": [np.nan, 0.3, 0.2],
                }
            ).to_csv(run_dir / "training_history.csv", index=False)

            row = analyze_multiseed.extract_training_summary(run_dir, fallback_model_seed=123)

        self.assertEqual(row["best_epoch"], 2)
        self.assertEqual(row["best_val_loss"], 0.8)
        self.assertEqual(row["outcome_cont_val_loss"], 0.4)
        self.assertEqual(row["outcome_bin_val_loss"], 0.2)
        self.assertFalse(any(pd.isna(row[name]) for name in (
            "best_val_loss", "outcome_cont_val_loss", "outcome_bin_val_loss"
        )))

    def test_flags_seed_123_as_training_outlier_without_removing_it(self):
        summary = pd.DataFrame(
            {
                "model_seed": [42, 123, 456, 789, 2026],
                "best_epoch": [121, 21, 114, 117, 119],
                "best_val_loss": [1.00, 1.90, 1.03, 0.98, 1.01],
                "outcome_cont_val_loss": [0.40, 1.10, 0.42, 0.39, 0.41],
                "outcome_bin_val_loss": [0.20, 0.55, 0.21, 0.19, 0.20],
            }
        )

        flagged = analyze_multiseed.flag_training_outliers(summary)

        seed_123 = flagged.loc[flagged["model_seed"] == 123].iloc[0]
        self.assertTrue(bool(seed_123["is_training_outlier"]))
        self.assertIn("best_epoch", seed_123["training_outlier_reasons"])
        self.assertEqual(len(flagged), 5)


class EdgeMetricTests(unittest.TestCase):
    def test_edge_statistics_and_sign_consistency_match_hand_calculation(self):
        edges = pd.DataFrame(
            {
                "model_seed": [1, 2, 3, 4, 1, 2, 3, 4],
                "modality": ["protein"] * 8,
                "source": ["p1"] * 4 + ["p2"] * 4,
                "target": ["AST"] * 8,
                "edge_weight": [2.0, 2.0, 2.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            }
        )
        ranked = analyze_multiseed.add_edge_ranks(edges)

        result = analyze_multiseed.aggregate_edge_statistics(ranked, model_seeds=[1, 2, 3, 4])

        p1 = result.loc[result["source"] == "p1"].iloc[0]
        self.assertAlmostEqual(p1["mean_effect"], 1.25)
        self.assertAlmostEqual(p1["median_effect"], 2.0)
        self.assertAlmostEqual(p1["mean_abs_effect"], 1.75)
        self.assertAlmostEqual(p1["std_effect"], float(np.std([2.0, 2.0, 2.0, -1.0])))
        self.assertAlmostEqual(p1["effect_cv"], p1["std_effect"] / 1.25)
        self.assertEqual(p1["sign_consistency"], 0.75)
        self.assertEqual(p1["positive_seed_count"], 3)
        self.assertEqual(p1["negative_seed_count"], 1)
        self.assertIn("model_seed_1", result.columns)
        p2 = result.loc[result["source"] == "p2"].iloc[0]
        self.assertTrue(math.isnan(p2["effect_cv"]))

    def test_pairwise_topk_metrics_match_hand_built_sets(self):
        edges = pd.DataFrame(
            {
                "model_seed": [1, 1, 1, 2, 2, 2],
                "modality": ["protein"] * 6,
                "source": ["a", "b", "c", "a", "b", "c"],
                "target": ["AST"] * 6,
                "edge_weight": [3.0, 2.0, 1.0, 3.0, 1.0, 2.0],
            }
        )
        ranked = analyze_multiseed.add_edge_ranks(edges)

        result, summary = analyze_multiseed.compute_pairwise_topk_stability(ranked, top_ks=(2,))

        row = result.iloc[0]
        self.assertEqual(row["intersection_count"], 1)
        self.assertAlmostEqual(row["jaccard_index"], 1 / 3)
        self.assertAlmostEqual(row["overlap_coefficient"], 1 / 2)
        self.assertAlmostEqual(summary.iloc[0]["mean_jaccard_index"], 1 / 3)

    def test_spearman_is_computed_separately_for_each_outcome(self):
        records = []
        for model_seed, ast_weights, alt_weights in (
            (1, [3.0, 2.0, 1.0], [3.0, 2.0, 1.0]),
            (2, [1.0, 2.0, 3.0], [6.0, 4.0, 2.0]),
        ):
            for source, weight in zip(["a", "b", "c"], ast_weights):
                records.append((model_seed, "protein", source, "AST", weight))
            for source, weight in zip(["a", "b", "c"], alt_weights):
                records.append((model_seed, "protein", source, "ALT", weight))
        ranked = analyze_multiseed.add_edge_ranks(
            pd.DataFrame(records, columns=["model_seed", "modality", "source", "target", "edge_weight"])
        )

        result, summary = analyze_multiseed.compute_pairwise_rank_correlation(ranked)

        ast = result.loc[result["target"] == "AST", "spearman_rank_correlation"].iloc[0]
        alt = result.loc[result["target"] == "ALT", "spearman_rank_correlation"].iloc[0]
        self.assertAlmostEqual(ast, -1.0)
        self.assertAlmostEqual(alt, 1.0)
        self.assertEqual(set(summary["target"]), {"AST", "ALT"})

    def test_ranked_consensus_is_deterministic_and_does_not_require_top100(self):
        aggregate = pd.DataFrame(
            {
                "modality": ["protein", "protein", "metabolite"],
                "source": ["stable_130", "strict", "variable"],
                "target": ["AST", "ALT", "AST"],
                "mean_abs_effect": [2.0, 3.0, 4.0],
                "std_effect": [0.05, 0.10, 4.0],
                "effect_cv": [0.025, 0.033, 1.0],
                "sign_consistency": [1.0, 1.0, 0.6],
                "mean_rank": [130.0, 20.0, 10.0],
                "median_rank": [130.0, 20.0, 10.0],
                "top100_frequency": [0.0, 0.8, 1.0],
            }
        )

        strict_a, ranked_a = analyze_multiseed.build_consensus_edges(aggregate)
        strict_b, ranked_b = analyze_multiseed.build_consensus_edges(aggregate.sample(frac=1, random_state=7))

        self.assertEqual(strict_a["source"].tolist(), ["strict"])
        self.assertIn("stable_130", ranked_a["source"].tolist())
        assert_frame_equal(ranked_a.reset_index(drop=True), ranked_b.reset_index(drop=True))

    def test_outcome_summary_separates_modalities_and_includes_decoder_norm(self):
        ranked = analyze_multiseed.add_edge_ranks(
            pd.DataFrame(
                {
                    "model_seed": [1, 1, 2, 2],
                    "modality": ["protein", "metabolite", "protein", "metabolite"],
                    "source": ["p1", "m1", "p1", "m1"],
                    "target": ["AST"] * 4,
                    "edge_weight": [3.0, 1.0, 3.0, 1.0],
                }
            )
        )
        strict = pd.DataFrame(
            {"modality": ["protein"], "source": ["p1"], "target": ["AST"]}
        )
        decoder_norms = pd.DataFrame(
            {
                "model_seed": [1, 2],
                "target": ["AST", "AST"],
                "outcome_decoder_norm": [2.0, 4.0],
            }
        )

        result = analyze_multiseed.build_outcome_edge_distribution(
            ranked, strict, decoder_norms
        ).iloc[0]

        self.assertAlmostEqual(result["protein_contribution"], 0.75)
        self.assertAlmostEqual(result["metabolite_contribution"], 0.25)
        self.assertEqual(result["consensus_edge_count"], 1)
        self.assertAlmostEqual(result["outcome_decoder_norm"], 3.0)


if __name__ == "__main__":
    unittest.main()
