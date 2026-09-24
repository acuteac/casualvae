import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analysis import annotate_metabolite_features


class MetaboliteDictionaryTests(unittest.TestCase):
    def test_parse_feature_code_extracts_field_and_instance(self):
        self.assertEqual(
            annotate_metabolite_features.parse_feature_code("p23477_i0"),
            (23477, 0),
        )
        self.assertEqual(
            annotate_metabolite_features.parse_feature_code("p23569_i1"),
            (23569, 1),
        )
        self.assertEqual(
            annotate_metabolite_features.parse_feature_code("Acetone"),
            (None, None),
        )

    def test_dictionary_maps_instances_and_retains_unmapped_codes(self):
        mapping = pd.DataFrame(
            {
                "field_id": [23477, 23569],
                "title": ["Acetone", "Cholesteryl esters in medium HDL"],
                "Group": ["Ketone bodies", "Lipoprotein subclasses"],
                "Subgroup": ["", "Medium HDL"],
            }
        )

        dictionary = annotate_metabolite_features.build_feature_dictionary(
            ["p23477_i0", "p23569_i0", "p23569_i1", "p99999_i0", "bad_code"],
            mapping,
            annotation_source="UK Biobank Resource 3543",
        ).set_index("feature_code")

        self.assertEqual(dictionary.loc["p23477_i0", "field_id"], 23477)
        self.assertEqual(dictionary.loc["p23477_i0", "instance"], 0)
        self.assertEqual(dictionary.loc["p23477_i0", "metabolite_name"], "Acetone")
        self.assertEqual(dictionary.loc["p23477_i0", "category"], "Ketone bodies")
        self.assertEqual(dictionary.loc["p23569_i1", "instance"], 1)
        self.assertEqual(
            dictionary.loc["p23569_i1", "metabolite_name"],
            "Cholesteryl esters in medium HDL",
        )
        self.assertEqual(dictionary.loc["p23569_i0", "annotation_status"], "mapped")
        self.assertEqual(dictionary.loc["p99999_i0", "metabolite_name"], "p99999_i0")
        self.assertEqual(dictionary.loc["p99999_i0", "annotation_status"], "unmapped")
        self.assertEqual(dictionary.loc["bad_code", "metabolite_name"], "bad_code")
        self.assertEqual(dictionary.loc["bad_code", "annotation_status"], "unmapped")

    def test_conflicting_source_titles_are_marked_ambiguous_not_guessed(self):
        mapping = pd.DataFrame(
            {
                "field_id": [23477, 23477],
                "title": ["Acetone", "Different title"],
                "Group": ["Ketone bodies", "Ketone bodies"],
                "Subgroup": ["", ""],
            }
        )

        dictionary = annotate_metabolite_features.build_feature_dictionary(
            ["p23477_i0"], mapping, annotation_source="source"
        ).iloc[0]

        self.assertEqual(dictionary["annotation_status"], "ambiguous")
        self.assertEqual(dictionary["metabolite_name"], "p23477_i0")

    def test_annotation_join_preserves_every_original_column_and_row(self):
        original = pd.DataFrame(
            {
                "direct_rank": [1, 2],
                "feature": ["p23477_i0", "p99999_i0"],
                "DirectScore": [0.003, 0.002],
            }
        )
        dictionary = pd.DataFrame(
            {
                "feature_code": ["p23477_i0", "p99999_i0"],
                "field_id": [23477, 99999],
                "instance": [0, 0],
                "metabolite_name": ["Acetone", "p99999_i0"],
                "metabolite_name_short": ["Acetone", "p99999_i0"],
                "annotation_source": ["source", ""],
                "annotation_status": ["mapped", "unmapped"],
                "category": ["Ketone bodies", ""],
                "subclass": ["", ""],
                "ukb_field_title": ["Acetone", ""],
            }
        )

        annotated = annotate_metabolite_features.annotate_table(original, dictionary)

        pd.testing.assert_frame_equal(annotated[original.columns], original)
        self.assertEqual(annotated["feature_code"].tolist(), original["feature"].tolist())
        self.assertEqual(len(annotated), len(original))

    def test_mixed_modality_table_only_annotates_metabolite_rows(self):
        original = pd.DataFrame(
            {
                "modality": ["Protein", "Metabolite"],
                "feature": ["p23477_i0", "p23477_i0"],
                "effect": [0.1, 0.2],
            }
        )
        dictionary = pd.DataFrame(
            {
                "feature_code": ["p23477_i0"],
                "field_id": [23477],
                "instance": [0],
                "metabolite_name": ["Acetone"],
                "metabolite_name_short": ["Acetone"],
                "annotation_source": ["source"],
                "annotation_status": ["mapped"],
                "category": ["Ketone bodies"],
                "subclass": [""],
                "ukb_field_title": ["Acetone"],
            }
        )

        annotated = annotate_metabolite_features.annotate_table(original, dictionary)

        self.assertTrue(pd.isna(annotated.loc[0, "feature_code"]))
        self.assertEqual(annotated.loc[1, "metabolite_name"], "Acetone")

    def test_official_snapshot_has_expected_unique_fields(self):
        snapshot = annotate_metabolite_features.load_mapping_source(
            ROOT / "artifacts/metadata/ukb_nightingale_biomarker_groups.tsv"
        )

        self.assertEqual(len(snapshot), 251)
        self.assertEqual(snapshot["field_id"].nunique(), 251)
        names = snapshot.set_index("field_id")["title"]
        self.assertEqual(names.loc[23477], "Acetone")
        self.assertEqual(names.loc[23475], "Acetate")
        self.assertEqual(names.loc[23450], "Docosahexaenoic acid")


class MetaboliteAnnotationCliTests(unittest.TestCase):
    def test_dictionary_output_cannot_overwrite_an_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            screening = root / "screening"
            screening.mkdir()
            mapping_source = root / "mapping.tsv"
            mapping_source.write_text(
                "field_id\ttitle\tGroup\tSubgroup\n"
                "23477\tAcetone\tKetone bodies\t\n",
                encoding="utf-8",
            )
            ranking = pd.DataFrame(
                {
                    "feature": ["p23477_i0"],
                    "DirectScore": [0.1],
                }
            )
            tiers = pd.DataFrame(
                {
                    "feature": ["p23477_i0"],
                    "primary_direct_rank": [1],
                    "primary_direct_score": [0.1],
                    "strongest_outcome": ["AST"],
                    "strongest_effect": [0.2],
                    "candidate_tier": ["Tier1"],
                    "sign_agreement_count": [3],
                    "sign_agreement_fraction": [1.0],
                    "all_edge_sign_agreement_fraction": [1.0],
                }
            )
            ranking_path = screening / "metabolite_direct_ranking.csv"
            ranking.to_csv(ranking_path, index=False)
            tiers.to_csv(screening / "metabolite_candidate_tiers.csv", index=False)
            before = ranking_path.read_bytes()

            with self.assertRaisesRegex(ValueError, "collides with an input"):
                annotate_metabolite_features.main(
                    [
                        "--screening_dir",
                        str(screening),
                        "--mapping_source",
                        str(mapping_source),
                        "--dictionary_out",
                        str(ranking_path),
                    ]
                )

            self.assertEqual(ranking_path.read_bytes(), before)

    def test_cli_generates_dictionary_annotated_tables_and_summary_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            screening = root / "screening"
            metadata = root / "metadata"
            screening.mkdir()
            metadata.mkdir()

            mapping_source = metadata / "ukb_nightingale.tsv"
            mapping_source.write_text(
                "field_id\ttitle\tGroup\tSubgroup\n"
                "23477\tAcetone\tKetone bodies\t\n"
                "23475\tAcetate\tKetone bodies\t\n"
                "23450\tDocosahexaenoic acid\tFatty acids\t\n",
                encoding="utf-8",
            )
            ranking = pd.DataFrame(
                {
                    "modality": ["Metabolite"] * 3,
                    "feature": ["p23477_i0", "p23475_i0", "p99999_i0"],
                    "direct_rank": [1, 2, 3],
                    "DirectScore": [0.0034, 0.0031, 0.0020],
                    "strongest_outcome": ["GGT", "AST", "ALT"],
                    "strongest_effect": [0.0066, 0.0088, -0.004],
                }
            )
            tiers = pd.DataFrame(
                {
                    "modality": ["Metabolite"] * 3,
                    "feature": ranking["feature"],
                    "primary_direct_rank": [1, 2, 3],
                    "primary_direct_score": ranking["DirectScore"],
                    "strongest_outcome": ranking["strongest_outcome"],
                    "strongest_effect": ranking["strongest_effect"],
                    "candidate_tier": ["Tier1", "Tier2", "Tier3"],
                    "sign_agreement_count": [3, 1, 0],
                    "sign_agreement_fraction": [1.0, 1 / 3, 0.0],
                    "all_edge_sign_agreement_fraction": [0.8, 0.6, 0.4],
                }
            )
            comparison = pd.DataFrame(
                {
                    "modality": ["Protein", "Metabolite"],
                    "feature": ["COQ7", "p23477_i0"],
                    "outcome": ["AST", "AST"],
                    "primary_effect": [0.1, 0.2],
                }
            )
            top_edges = pd.DataFrame(
                {
                    "modality": ["Protein", "Metabolite"],
                    "feature": ["COQ7", "p23475_i0"],
                    "outcome": ["AST", "AST"],
                    "effect": [0.1, 0.2],
                }
            )
            originals = {
                "metabolite_direct_ranking.csv": ranking,
                "metabolite_candidate_tiers.csv": tiers,
                "cross_model_direct_effect_comparison.csv": comparison,
                "top_direct_edges_by_outcome.csv": top_edges,
            }
            for filename, frame in originals.items():
                frame.to_csv(screening / filename, index=False)
            original_bytes = {
                filename: (screening / filename).read_bytes() for filename in originals
            }

            dictionary_out = metadata / "metabolite_feature_dictionary.csv"
            exit_code = annotate_metabolite_features.main(
                [
                    "--screening_dir",
                    str(screening),
                    "--mapping_source",
                    str(mapping_source),
                    "--dictionary_out",
                    str(dictionary_out),
                    "--annotation_source",
                    "UK Biobank Resource 3543",
                ]
            )

            self.assertEqual(exit_code, 0)
            for filename, before in original_bytes.items():
                self.assertEqual((screening / filename).read_bytes(), before)
            dictionary = pd.read_csv(dictionary_out)
            self.assertEqual(len(dictionary), 3)
            self.assertEqual(dictionary["feature_code"].nunique(), 3)
            self.assertEqual(dictionary["annotation_status"].value_counts().to_dict(), {"mapped": 2, "unmapped": 1})

            annotated_ranking = pd.read_csv(
                screening / "metabolite_direct_ranking_annotated.csv"
            )
            annotated_tiers = pd.read_csv(
                screening / "metabolite_candidate_tiers_annotated.csv"
            )
            pd.testing.assert_frame_equal(
                annotated_ranking[ranking.columns], ranking, check_exact=True
            )
            pd.testing.assert_frame_equal(
                annotated_tiers[tiers.columns], tiers, check_exact=True
            )
            annotated_comparison = pd.read_csv(
                screening / "cross_model_direct_effect_comparison_annotated.csv"
            )
            self.assertEqual(len(annotated_comparison), len(comparison))
            self.assertTrue(pd.isna(annotated_comparison.loc[0, "feature_code"]))
            self.assertEqual(annotated_comparison.loc[1, "metabolite_name"], "Acetone")

            candidate_summary = pd.read_csv(
                screening / "metabolite_candidate_annotation_summary.csv"
            )
            self.assertEqual(candidate_summary["primary_rank"].tolist(), [1, 2, 3])
            self.assertEqual(candidate_summary.loc[0, "metabolite_name"], "Acetone")
            summary = json.loads(
                (screening / "metabolite_annotation_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["total_metabolite_features"], 3)
            self.assertEqual(summary["mapped_count"], 2)
            self.assertEqual(summary["unmapped_count"], 1)
            self.assertEqual(summary["ambiguous_count"], 0)
            self.assertAlmostEqual(summary["mapping_rate"], 2 / 3)
            self.assertEqual(summary["top10_primary_metabolites"][0]["metabolite_name"], "Acetone")

            generated_paths = [
                dictionary_out,
                screening / "metabolite_direct_ranking_annotated.csv",
                screening / "metabolite_candidate_tiers_annotated.csv",
                screening / "cross_model_direct_effect_comparison_annotated.csv",
                screening / "top_direct_edges_by_outcome_annotated.csv",
                screening / "metabolite_candidate_annotation_summary.csv",
                screening / "metabolite_annotation_summary.json",
            ]
            first_run = {path: path.read_bytes() for path in generated_paths}
            self.assertEqual(
                annotate_metabolite_features.main(
                    [
                        "--screening_dir",
                        str(screening),
                        "--mapping_source",
                        str(mapping_source),
                        "--dictionary_out",
                        str(dictionary_out),
                        "--annotation_source",
                        "UK Biobank Resource 3543",
                    ]
                ),
                0,
            )
            self.assertEqual(
                {path: path.read_bytes() for path in generated_paths},
                first_run,
            )


if __name__ == "__main__":
    unittest.main()
