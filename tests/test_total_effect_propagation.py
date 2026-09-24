import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from scripts.analysis import total_effect_propagation
except ImportError:
    total_effect_propagation = None


class LatentPropagationTests(unittest.TestCase):
    def test_total_propagation_is_computed_in_latent_space(self):
        self.assertIsNotNone(total_effect_propagation)
        adjacency = pd.DataFrame(
            [[0.0, 0.2], [0.3, 0.0]],
            index=["L1", "L2"],
            columns=["L1", "L2"],
        )

        propagation, diagnostics = (
            total_effect_propagation.compute_latent_total_propagation(adjacency)
        )

        expected = np.array(
            [
                [0.0638297872340425, 0.2127659574468085],
                [0.3191489361702128, 0.0638297872340425],
            ]
        )
        np.testing.assert_allclose(propagation.to_numpy(), expected, rtol=1e-12)
        self.assertAlmostEqual(diagnostics["spectral_radius_A"], np.sqrt(0.06))
        self.assertTrue(diagnostics["stable_spectral_radius"])
        self.assertGreaterEqual(diagnostics["condition_number_I_minus_A"], 1.0)

    def test_projection_returns_only_outcome_by_molecular_blocks(self):
        self.assertIsNotNone(total_effect_propagation)
        adjacency = pd.DataFrame(
            [[0.0, 0.2], [0.3, 0.0]],
            index=["L1", "L2"],
            columns=["L1", "L2"],
        )
        decoder = pd.DataFrame(
            [[1.0, 0.0], [0.5, 0.0], [0.0, 1.0], [0.0, 1.0]],
            index=["P1", "P2", "M1", "AST"],
            columns=["L1", "L2"],
        )
        total_latent, _ = (
            total_effect_propagation.compute_latent_total_propagation(adjacency)
        )

        blocks = total_effect_propagation.project_molecular_outcome_blocks(
            adjacency,
            total_latent,
            decoder,
            protein_features=["P1", "P2"],
            metabolite_features=["M1"],
            outcomes=["AST"],
        )

        self.assertEqual(blocks["Protein"]["direct"].shape, (1, 2))
        self.assertEqual(blocks["Metabolite"]["total"].shape, (1, 1))
        self.assertAlmostEqual(blocks["Protein"]["direct"].loc["AST", "P1"], 0.3)
        self.assertAlmostEqual(
            blocks["Protein"]["total"].loc["AST", "P1"],
            0.3191489361702128,
        )
        self.assertAlmostEqual(
            blocks["Metabolite"]["total"].loc["AST", "M1"],
            0.0638297872340425,
        )


class EffectRankingTests(unittest.TestCase):
    def test_rank_and_patterns_follow_absolute_direct_and_total_effects(self):
        self.assertIsNotNone(total_effect_propagation)
        direct = pd.DataFrame(
            [[4.0, 3.0, 2.0, 1.0]],
            index=["AST"],
            columns=["f1", "f2", "f3", "f4"],
        )
        total = pd.DataFrame(
            [[4.0, 0.5, 5.0, -3.0]],
            index=["AST"],
            columns=["f1", "f2", "f3", "f4"],
        )

        edges = total_effect_propagation.build_model_edge_table(
            direct, total, "Protein", high_k=1
        ).set_index("feature")

        self.assertEqual(edges.loc["f1", "direct_outcome_rank"], 1)
        self.assertEqual(edges.loc["f1", "total_outcome_rank"], 2)
        self.assertEqual(edges.loc["f1", "effect_pattern"], "attenuated")
        self.assertEqual(edges.loc["f3", "direct_outcome_rank"], 3)
        self.assertEqual(edges.loc["f3", "total_outcome_rank"], 1)
        self.assertEqual(edges.loc["f3", "rank_shift"], 2)
        self.assertEqual(edges.loc["f3", "effect_pattern"], "indirect_amplified")
        self.assertTrue(edges.loc["f4", "direction_reversal"])
        np.testing.assert_allclose(
            edges["total_effect"],
            edges["direct_effect"] + edges["indirect_effect"],
        )


class CrossModelTotalEffectTests(unittest.TestCase):
    def setUp(self):
        direct = pd.DataFrame(
            [[4.0, 3.0, 2.0, 1.0]],
            index=["AST"],
            columns=["f1", "f2", "f3", "f4"],
        )
        totals = {
            "primary": [4.0, 0.5, 5.0, -3.0],
            "variable456": [1.0, 2.0, 4.0, -3.0],
            "variable789": [-5.0, 4.0, 3.0, 2.0],
            "fixed789": [6.0, 5.0, -4.0, 1.0],
        }
        self.edges = {
            label: total_effect_propagation.build_model_edge_table(
                direct,
                pd.DataFrame([values], index=["AST"], columns=direct.columns),
                "Protein",
                high_k=1,
            )
            for label, values in totals.items()
        }
        self.roles = {
            "primary": "primary",
            "variable456": "replication",
            "variable789": "replication",
            "fixed789": "sensitivity",
        }

    def test_cross_model_support_counts_use_the_same_feature_outcome_edge(self):
        self.assertTrue(
            hasattr(total_effect_propagation, "build_cross_model_comparison")
        )
        comparison = total_effect_propagation.build_cross_model_comparison(
            self.edges, self.roles, top_ks=(1, 2, 3), high_k=1
        ).set_index("feature")

        f3 = comparison.loc["f3"]
        self.assertEqual(f3["primary_total_effect"], 5.0)
        self.assertEqual(f3["primary_direct_outcome_rank"], 3)
        self.assertEqual(f3["primary_total_outcome_rank"], 1)
        self.assertEqual(f3["total_sign_agreement_count"], 2)
        self.assertAlmostEqual(f3["total_sign_agreement_fraction"], 2 / 3)
        self.assertEqual(f3["total_top1_support_count"], 2)
        self.assertEqual(f3["total_top3_support_count"], 4)
        self.assertEqual(f3["rank_shift_positive_support_count"], 2)
        self.assertEqual(f3["indirect_amplified_support_count"], 2)

    def test_total_stability_compares_primary_with_each_other_model(self):
        self.assertTrue(hasattr(total_effect_propagation, "build_total_stability"))
        stability = total_effect_propagation.build_total_stability(
            self.edges, self.roles, top_ks=(1, 2, 3)
        )

        self.assertEqual(len(stability), 3)
        row = stability.loc[
            stability["comparison_model"] == "variable456"
        ].iloc[0]
        self.assertAlmostEqual(row["total_rank_spearman"], 0.4)
        self.assertEqual(row["total_top1_intersection"], 1)
        self.assertEqual(row["total_top1_jaccard"], 1.0)
        self.assertEqual(row["comparison_role"], "replication")

    def test_multi_outcome_summary_counts_total_and_pattern_recurrence(self):
        self.assertTrue(
            hasattr(total_effect_propagation, "build_multi_outcome_summary")
        )
        ast = self.edges["primary"].copy()
        alt = ast.copy()
        alt["outcome"] = "ALT"
        alt.loc[alt["feature"] == "f1", "total_outcome_rank"] = 1
        alt.loc[alt["feature"] == "f3", "total_outcome_rank"] = 2
        alt.loc[alt["feature"] == "f1", "effect_pattern"] = "direct_and_total_high"
        alt.loc[alt["feature"] == "f4", "direction_reversal"] = False

        summary = total_effect_propagation.build_multi_outcome_summary(
            pd.concat([ast, alt], ignore_index=True), top_ks=(1, 2, 3)
        ).set_index("feature")

        self.assertEqual(summary.loc["f3", "n_outcomes_total_top1"], 1)
        self.assertEqual(summary.loc["f3", "outcomes_total_top2"], "ALT;AST")
        self.assertEqual(summary.loc["f3", "n_outcomes_indirect_amplified"], 2)
        self.assertEqual(summary.loc["f4", "n_outcomes_direction_reversal"], 1)

    def test_direct_reference_requires_exact_edges_and_integral_ranks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "direct.csv"
            rows = []
            for row in self.edges["primary"].itertuples():
                item = {
                    "modality": row.modality,
                    "outcome": row.outcome,
                    "feature": row.feature,
                }
                for label, edges in self.edges.items():
                    match = edges.loc[edges["feature"] == row.feature].iloc[0]
                    item[f"{label}_effect"] = match["direct_effect"]
                    item[f"{label}_outcome_rank"] = match["direct_outcome_rank"]
                rows.append(item)
            extra = rows[0].copy()
            extra["feature"] = "unexpected_feature"
            pd.DataFrame([*rows, extra]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "edge keys differ"):
                total_effect_propagation._validate_against_direct_comparison(
                    self.edges, path
                )

            non_integral = pd.DataFrame(rows)
            non_integral["primary_outcome_rank"] = non_integral[
                "primary_outcome_rank"
            ].astype(float)
            non_integral.loc[0, "primary_outcome_rank"] = 1.5
            non_integral.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "integer ranks"):
                total_effect_propagation._validate_against_direct_comparison(
                    self.edges, path
                )

    def test_pattern_summary_reconciles_every_outcome_group(self):
        summary = total_effect_propagation._pattern_summary(self.edges["primary"])
        ast = summary["by_modality_outcome"][0]

        self.assertIn("neither_high", ast)
        self.assertEqual(ast["neither_high"], 2)
        self.assertEqual(
            ast["direct_and_total_high"]
            + ast["indirect_amplified"]
            + ast["attenuated"]
            + ast["neither_high"],
            4,
        )


class VisualizationNamingTests(unittest.TestCase):
    def test_distinct_outcomes_that_sanitize_alike_get_distinct_filenames(self):
        slash = total_effect_propagation._outcome_filename("A/B")
        space = total_effect_propagation._outcome_filename("A B")

        self.assertNotEqual(slash, space)
        self.assertEqual(
            total_effect_propagation._outcome_filename("AST"), "outcome_AST.html"
        )


class TotalEffectCliTests(unittest.TestCase):
    def test_cli_writes_analysis_and_outcome_visualizations(self):
        self.assertTrue(hasattr(total_effect_propagation, "main"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directories = {
                "primary": root / "fixed_split" / "model_seed_456",
                "variable456": root / "variable_split" / "seed_456",
                "variable789": root / "variable_split" / "seed_789",
                "fixed789": root / "fixed_split" / "model_seed_789",
            }
            factors = {
                "primary": 1.0,
                "variable456": 0.8,
                "variable789": 0.6,
                "fixed789": 0.4,
            }
            latent = ["L1", "L2"]
            features = ["P1", "P2", "M1", "M2", "AST", "ALT"]
            decoder = pd.DataFrame(
                [
                    [1.0, 0.0],
                    [0.5, 0.0],
                    [0.0, 1.0],
                    [0.2, 0.8],
                    [0.0, 1.0],
                    [1.0, 0.0],
                ],
                index=features,
                columns=latent,
            )
            reference_rows = []
            direct_by_model = {}
            for label, directory in directories.items():
                directory.mkdir(parents=True)
                factor = factors[label]
                adjacency = pd.DataFrame(
                    [[0.0, 0.2 * factor], [0.3 * factor, 0.0]],
                    index=latent,
                    columns=latent,
                )
                projected = decoder.to_numpy() @ adjacency.to_numpy() @ decoder.to_numpy().T
                projected = pd.DataFrame(projected, index=features, columns=features)
                protein = projected.loc[["AST", "ALT"], ["P1", "P2"]]
                metabolite = projected.loc[["AST", "ALT"], ["M1", "M2"]]
                adjacency.to_csv(directory / "latent_causal_A.csv")
                decoder.to_csv(directory / "decoder_projection_D.csv")
                protein.to_csv(directory / "G_protein_to_outcome.csv")
                metabolite.to_csv(directory / "G_metabolite_to_outcome.csv")
                direct_by_model[label] = {
                    "Protein": protein,
                    "Metabolite": metabolite,
                }

            for modality in ["Protein", "Metabolite"]:
                for outcome in ["AST", "ALT"]:
                    feature_names = direct_by_model["primary"][modality].columns.tolist()
                    for feature in feature_names:
                        row = {
                            "modality": modality,
                            "outcome": outcome,
                            "feature": feature,
                        }
                        for label in directories:
                            matrix = direct_by_model[label][modality]
                            ordered = sorted(
                                feature_names,
                                key=lambda name: (-abs(matrix.loc[outcome, name]), name),
                            )
                            row[f"{label}_effect"] = matrix.loc[outcome, feature]
                            row[f"{label}_outcome_rank"] = ordered.index(feature) + 1
                        reference_rows.append(row)
            direct_reference = root / "direct_comparison.csv"
            pd.DataFrame(reference_rows).to_csv(direct_reference, index=False)

            dictionary = root / "metabolite_dictionary.csv"
            pd.DataFrame(
                {
                    "feature_code": ["M1", "M2"],
                    "metabolite_name": ["Metabolite one", "Metabolite two"],
                }
            ).to_csv(dictionary, index=False)
            out_dir = root / "analysis"
            visual_dir = root / "visual"

            exit_code = total_effect_propagation.main(
                [
                    "--primary_dir",
                    str(directories["primary"]),
                    "--replication_dirs",
                    str(directories["variable456"]),
                    str(directories["variable789"]),
                    "--sensitivity_dirs",
                    str(directories["fixed789"]),
                    "--direct_comparison",
                    str(direct_reference),
                    "--metabolite_dictionary",
                    str(dictionary),
                    "--out_dir",
                    str(out_dir),
                    "--visualization_dir",
                    str(visual_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            expected_files = {
                "total_effect_primary.csv",
                "total_effect_cross_model_comparison.csv",
                "total_effect_top_edges.csv",
                "direct_vs_total_candidates.csv",
                "total_effect_stability_summary.csv",
                "multi_outcome_total_effect_summary.csv",
                "total_effect_summary.json",
            }
            self.assertEqual(expected_files, {path.name for path in out_dir.iterdir()})
            self.assertEqual(
                {"index.html", "outcome_AST.html", "outcome_ALT.html"},
                {path.name for path in visual_dir.iterdir()},
            )
            primary = pd.read_csv(out_dir / "total_effect_primary.csv")
            comparison = pd.read_csv(
                out_dir / "total_effect_cross_model_comparison.csv"
            )
            self.assertEqual(len(primary), 8)
            self.assertEqual(len(comparison), 8)
            self.assertFalse(primary.isna().any().any())
            self.assertEqual(
                primary.loc[primary["feature"] == "M1", "metabolite_name"].iloc[0],
                "Metabolite one",
            )
            summary = json.loads(
                (out_dir / "total_effect_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["primary_edge_count"], 8)
            self.assertEqual(summary["outcome_count"], 2)
            self.assertEqual(summary["feature_counts"]["Protein"], 2)
            self.assertTrue(summary["models"]["primary"]["stable_spectral_radius"])
            index_text = (visual_dir / "index.html").read_text(encoding="utf-8")
            outcome_text = (visual_dir / "outcome_AST.html").read_text(
                encoding="utf-8"
            )
            self.assertIn("总传播效应排名的跨模型稳定性", index_text)
            self.assertIn("多 Outcome 总效应候选", index_text)
            self.assertIn(
                "用于观察同一 Outcome 的总效应排序在不同模型间是否一致",
                index_text,
            )
            self.assertIn(
                "统计同一特征在多个 Outcome 中重复进入 Total Top-K 的情况",
                index_text,
            )
            self.assertIn("蛋白质总效应 Top20", outcome_text)
            self.assertIn("代谢物总效应 Top20", outcome_text)
            self.assertIn("直接效应排名与总效应排名", outcome_text)
            self.assertIn("间接放大候选", outcome_text)
            self.assertIn("衰减候选", outcome_text)
            self.assertIn("方向翻转候选", outcome_text)
            self.assertIn(
                "按总效应绝对值从高到低列出该 Outcome 的前 20 个蛋白质",
                outcome_text,
            )
            self.assertIn(
                "按总效应绝对值从高到低列出该 Outcome 的前 20 个代谢物",
                outcome_text,
            )
            self.assertIn(
                "横轴为直接效应排名，纵轴为总效应排名", outcome_text
            )
            self.assertIn(
                "直接效应排名大于 100、总效应排名进入前 100",
                outcome_text,
            )
            self.assertIn(
                "直接效应排名在前 100、总效应排名跌出前 100",
                outcome_text,
            )
            self.assertIn("直接效应与总效应符号相反", outcome_text)
            self.assertIn("Metabolite one [M1]", index_text)
            self.assertIn("Metabolite one [M1]", outcome_text)
            self.assertNotIn("metabolite_name", index_text)
            self.assertNotIn("metabolite_name", outcome_text)


if __name__ == "__main__":
    unittest.main()
