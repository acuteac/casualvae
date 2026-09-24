import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analysis import direct_effect_screening


class OutcomeRankingTests(unittest.TestCase):
    def test_matrix_to_edges_ranks_absolute_effect_with_feature_tie_break(self):
        matrix = pd.DataFrame(
            {
                "z_feature": [2.0, 1.0],
                "a_feature": [-2.0, -3.0],
                "m_feature": [0.5, -1.0],
            },
            index=["AST", "ALT"],
        )

        edges = direct_effect_screening.matrix_to_edge_table(matrix, "Protein")

        ast = edges.loc[edges["outcome"] == "AST"]
        alt = edges.loc[edges["outcome"] == "ALT"]
        self.assertEqual(ast["feature"].tolist(), ["a_feature", "z_feature", "m_feature"])
        self.assertEqual(ast["outcome_rank"].tolist(), [1, 2, 3])
        self.assertEqual(ast.iloc[0]["effect"], -2.0)
        self.assertEqual(alt["feature"].tolist(), ["a_feature", "m_feature", "z_feature"])
        self.assertEqual(alt["outcome_rank"].tolist(), [1, 2, 3])

    def test_loading_rejects_non_finite_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "G.csv"
            pd.DataFrame(
                {"P1": [1.0, float("inf")]}, index=["AST", "ALT"]
            ).to_csv(path)

            with self.assertRaisesRegex(ValueError, "NaN or infinite"):
                direct_effect_screening.load_effect_matrix(path)

    def test_matrix_validation_rejects_null_or_blank_labels(self):
        cases = [
            pd.DataFrame({"P1": [1.0]}, index=[None]),
            pd.DataFrame({"   ": [1.0]}, index=["AST"]),
        ]
        for matrix in cases:
            with self.subTest(matrix=matrix):
                with self.assertRaisesRegex(ValueError, "null or blank"):
                    direct_effect_screening.matrix_to_edge_table(matrix, "Protein")

    def test_loading_rejects_blank_or_duplicate_raw_csv_feature_headers(self):
        cases = {
            "blank": ",P1,   \nAST,1.0,2.0\n",
            "duplicate": ",P1,P1\nAST,1.0,2.0\n",
        }
        with tempfile.TemporaryDirectory() as tmp:
            for name, contents in cases.items():
                with self.subTest(name=name):
                    path = Path(tmp) / f"{name}.csv"
                    path.write_text(contents, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "feature header"):
                        direct_effect_screening.load_effect_matrix(path)


class CrossModelOutcomeTests(unittest.TestCase):
    def setUp(self):
        matrices = {
            "primary": pd.DataFrame(
                {"f1": [4.0], "f2": [-3.0], "f3": [2.0], "f4": [1.0]},
                index=["AST"],
            ),
            "variable456": pd.DataFrame(
                {"f1": [3.0], "f2": [-4.0], "f3": [1.0], "f4": [-2.0]},
                index=["AST"],
            ),
            "variable789": pd.DataFrame(
                {"f1": [-4.0], "f2": [-2.0], "f3": [3.0], "f4": [1.0]},
                index=["AST"],
            ),
            "fixed789": pd.DataFrame(
                {"f1": [3.0], "f2": [2.0], "f3": [4.0], "f4": [1.0]},
                index=["AST"],
            ),
        }
        self.edges = {
            label: direct_effect_screening.matrix_to_edge_table(matrix, "Protein")
            for label, matrix in matrices.items()
        }
        self.roles = {
            "primary": "primary",
            "variable456": "replication",
            "variable789": "replication",
            "fixed789": "sensitivity",
        }

    def test_cross_model_comparison_uses_same_edge_and_outcome_specific_ranks(self):
        comparison = direct_effect_screening.build_cross_model_comparison(
            self.edges,
            self.roles,
            top_ks=(1, 2, 3),
        )

        f1 = comparison.loc[comparison["feature"] == "f1"].iloc[0]
        self.assertEqual(f1["primary_effect"], 4.0)
        self.assertEqual(f1["variable456_effect"], 3.0)
        self.assertEqual(f1["variable789_effect"], -4.0)
        self.assertEqual(f1["fixed789_effect"], 3.0)
        self.assertEqual(f1["primary_outcome_rank"], 1)
        self.assertEqual(f1["variable456_outcome_rank"], 2)
        self.assertEqual(f1["variable789_outcome_rank"], 1)
        self.assertEqual(f1["fixed789_outcome_rank"], 2)
        self.assertEqual(f1["sign_agreement_count"], 2)
        self.assertAlmostEqual(f1["sign_agreement_fraction"], 2 / 3)
        self.assertEqual(f1["top1_support_count"], 2)
        self.assertEqual(f1["top2_support_count"], 4)
        self.assertEqual(f1["top3_support_count"], 4)
        self.assertEqual(f1["median_outcome_rank"], 1.5)
        self.assertEqual(f1["mean_outcome_rank"], 1.5)
        self.assertEqual(f1["best_outcome_rank"], 1)
        self.assertEqual(f1["worst_outcome_rank"], 2)

    def test_outcome_stability_is_primary_vs_each_other_model(self):
        stability = direct_effect_screening.build_outcome_stability(
            self.edges,
            self.roles,
            top_ks=(1, 2, 3),
        )

        self.assertEqual(len(stability), 3)
        variable456 = stability.loc[
            stability["comparison_model"] == "variable456"
        ].iloc[0]
        self.assertAlmostEqual(variable456["spearman_rank_correlation"], 0.6)
        self.assertEqual(variable456["top1_intersection"], 0)
        self.assertEqual(variable456["top1_jaccard"], 0.0)
        self.assertEqual(variable456["top2_intersection"], 2)
        self.assertEqual(variable456["top2_jaccard"], 1.0)
        self.assertEqual(variable456["comparison_role"], "replication")

    def test_multi_outcome_summary_counts_repeated_primary_top_edges(self):
        edges = pd.DataFrame(
            {
                "modality": ["Protein"] * 6,
                "outcome": ["AST"] * 3 + ["ALT"] * 3,
                "feature": ["A", "B", "C", "A", "C", "D"],
                "effect": [3.0, 2.0, 1.0, 4.0, 2.0, 1.0],
                "abs_effect": [3.0, 2.0, 1.0, 4.0, 2.0, 1.0],
                "outcome_rank": [1, 2, 3, 1, 2, 3],
            }
        )

        summary = direct_effect_screening.build_multi_outcome_summary(
            edges, top_ks=(1, 2, 3)
        ).set_index("feature")

        self.assertEqual(summary.loc["A", "n_outcomes_top1"], 2)
        self.assertEqual(summary.loc["A", "outcomes_top1"], "ALT;AST")
        self.assertEqual(summary.loc["C", "n_outcomes_top2"], 1)
        self.assertEqual(summary.loc["C", "n_outcomes_top3"], 2)
        self.assertEqual(summary.loc["D", "n_outcomes_top2"], 0)


class OutcomeCenteredCliTests(unittest.TestCase):
    def test_model_specs_require_two_replications_one_sensitivity_and_distinct_paths(self):
        primary = Path("primary")
        replications = [Path("replication_1"), Path("replication_2")]
        sensitivity = [Path("sensitivity")]

        with self.assertRaisesRegex(ValueError, "exactly two replication"):
            direct_effect_screening._model_specs(
                primary, replications[:1], sensitivity
            )
        with self.assertRaisesRegex(ValueError, "exactly two replication"):
            direct_effect_screening._model_specs(
                primary, [*replications, Path("replication_3")], sensitivity
            )
        with self.assertRaisesRegex(ValueError, "exactly one sensitivity"):
            direct_effect_screening._model_specs(primary, replications, [])
        with self.assertRaisesRegex(ValueError, "distinct directories"):
            direct_effect_screening._model_specs(
                primary, [primary, replications[1]], sensitivity
            )

    def test_cli_writes_only_outcome_centered_outputs_with_metabolite_names(self):
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
                "variable789": -0.6,
                "fixed789": 0.5,
            }
            for label, directory in directories.items():
                directory.mkdir(parents=True)
                pd.DataFrame(
                    {
                        "P1": [3.0 * factors[label], -1.0],
                        "P2": [1.0, 2.0 * factors[label]],
                        "P3": [0.5, 0.2],
                    },
                    index=["AST", "ALT"],
                ).to_csv(directory / "G_protein_to_outcome.csv")
                pd.DataFrame(
                    {
                        "p23477_i0": [1.0 * factors[label], -2.0],
                        "p23475_i0": [0.4, 0.25 * factors[label]],
                    },
                    index=["AST", "ALT"],
                ).to_csv(directory / "G_metabolite_to_outcome.csv")

            dictionary = root / "metabolite_feature_dictionary.csv"
            pd.DataFrame(
                {
                    "feature_code": ["p23477_i0", "p23475_i0"],
                    "metabolite_name": ["Acetone", "Acetate"],
                }
            ).to_csv(dictionary, index=False)

            out_dir = root / "analysis"
            exit_code = direct_effect_screening.main(
                [
                    "--primary_dir",
                    str(directories["primary"]),
                    "--replication_dirs",
                    str(directories["variable456"]),
                    str(directories["variable789"]),
                    "--sensitivity_dirs",
                    str(directories["fixed789"]),
                    "--out_dir",
                    str(out_dir),
                    "--metabolite_dictionary",
                    str(dictionary),
                    "--top_edges_per_outcome",
                    "2",
                ]
            )

            expected = {
                "outcome_specific_direct_edges.csv",
                "outcome_specific_cross_model_comparison.csv",
                "outcome_specific_top_edges.csv",
                "outcome_specific_stability_summary.csv",
                "multi_outcome_feature_summary.csv",
                "direct_effect_summary.json",
            }
            self.assertEqual(exit_code, 0)
            self.assertEqual(expected, {path.name for path in out_dir.iterdir()})

            direct_edges = pd.read_csv(out_dir / "outcome_specific_direct_edges.csv")
            comparison = pd.read_csv(
                out_dir / "outcome_specific_cross_model_comparison.csv"
            )
            stability = pd.read_csv(
                out_dir / "outcome_specific_stability_summary.csv"
            )
            self.assertEqual(len(direct_edges), 10)
            self.assertEqual(len(comparison), 10)
            self.assertEqual(len(stability), 12)
            self.assertNotIn("DirectScore", direct_edges.columns)
            self.assertNotIn("candidate_tier", comparison.columns)
            self.assertTrue(
                {"top20_support_count", "top50_support_count", "top100_support_count"}
                .issubset(comparison.columns)
            )
            acetone = direct_edges.loc[
                (direct_edges["modality"] == "Metabolite")
                & (direct_edges["feature"] == "p23477_i0")
                & (direct_edges["outcome"] == "AST")
            ].iloc[0]
            self.assertEqual(acetone["metabolite_name"], "Acetone")
            self.assertEqual(acetone["effect"], 1.0)
            self.assertEqual(acetone["outcome_rank"], 1)
            self.assertFalse(comparison.isna().any().any())
            numeric = comparison.select_dtypes(include=[np.number]).to_numpy()
            self.assertTrue(np.isfinite(numeric).all())

            summary = json.loads(
                (out_dir / "direct_effect_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["analysis_unit"], "feature_to_outcome_edge")
            self.assertEqual(summary["outcome_count"], 2)
            self.assertEqual(summary["feature_counts"]["Protein"], 3)
            self.assertEqual(summary["feature_counts"]["Metabolite"], 2)
            self.assertNotIn("direct_score_definition", summary)
            self.assertNotIn("tier_counts", summary)


if __name__ == "__main__":
    unittest.main()
