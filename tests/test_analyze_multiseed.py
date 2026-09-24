import os
import sys
import json
import pandas as pd
import tempfile
import unittest
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.analysis import analyze_multiseed

class TestAnalyzeMultiseed(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.base_out_dir = os.path.join(self.test_dir.name, "outputs", "multiseed")
        os.makedirs(self.base_out_dir)

        seeds = [42, 123, 456, 789, 2026]

        for i, seed in enumerate(seeds):
            seed_dir = os.path.join(self.base_out_dir, f"seed_{seed}")
            os.makedirs(seed_dir)

            # mock run_summary.json
            with open(os.path.join(seed_dir, "run_summary.json"), 'w') as f:
                json.dump({
                    "best_epoch": 10 + i,
                    "best_tau": 0.5,
                    "split_seed": 42,
                    "model_seed": seed,
                }, f)

            # mock training_history.csv
            pd.DataFrame({
                "stage": ["joint"],
                "epoch": [10 + i],
                "val_loss": [1.0 - (i * 0.1)],
                "val_outcome_cont_loss": [0.5],
                "val_outcome_bin_loss": [0.3]
            }).to_csv(os.path.join(seed_dir, "training_history.csv"), index=False)

            # mock G_protein_to_outcome.csv
            # Protein 1 has mostly positive weights
            # Protein 2 has mixed weights
            p_edges = pd.DataFrame({
                "source": ["Protein_1", "Protein_2", "Protein_3"],
                "target": ["Outcome_A", "Outcome_A", "Outcome_A"],
                "edge_weight": [2.0 if i < 4 else -0.5,
                                1.0 if i % 2 == 0 else -1.0,
                                0.1 * i]
            })
            p_edges.to_csv(os.path.join(seed_dir, "G_protein_to_outcome.csv"), index=False)

            # mock G_metabolite_to_outcome.csv
            m_edges = pd.DataFrame({
                "source": ["Metabolite_1"],
                "target": ["Outcome_B"],
                "edge_weight": [3.0]  # Very stable
            })
            m_edges.to_csv(os.path.join(seed_dir, "G_metabolite_to_outcome.csv"), index=False)

    def tearDown(self):
        self.test_dir.cleanup()
        return None

    def test_main_logic(self):
        analysis_dir = os.path.join(self.test_dir.name, "analysis")
        analyze_multiseed.analyze_experiment(
            self.base_out_dir,
            analysis_dir,
            experiment_name="fixed_split_experiment",
            fallback_split_seed=42,
        )

        expected_outputs = {
            "multiseed_training_summary.csv",
            "multiseed_edge_stability.csv",
            "pairwise_topk_stability.csv",
            "pairwise_rank_correlation.csv",
            "outcome_edge_distribution.csv",
            "consensus_edges_strict.csv",
            "consensus_edges_ranked.csv",
            "consensus_outcome_edges.csv",
        }
        self.assertTrue(expected_outputs.issubset(set(os.listdir(analysis_dir))))

        summary = pd.read_csv(os.path.join(analysis_dir, "multiseed_training_summary.csv"))
        self.assertFalse(summary[[
            "best_val_loss", "outcome_cont_val_loss", "outcome_bin_val_loss"
        ]].isna().any().any())

        # check stability output
        stability = pd.read_csv(os.path.join(analysis_dir, "multiseed_edge_stability.csv"))

        # Metabolite_1 -> Outcome_B should have sign_consistency = 1.0 and top100_frequency = 1.0
        metab1 = stability[stability["source"] == "Metabolite_1"].iloc[0]
        self.assertEqual(metab1["sign_consistency"], 1.0)
        self.assertEqual(metab1["top100_frequency"], 1.0)

        # Protein_1 -> Outcome_A: 4 positive, 1 negative. sign_consistency = 0.8
        prot1 = stability[stability["source"] == "Protein_1"].iloc[0]
        self.assertEqual(prot1["sign_consistency"], 0.8)

        # Check consensus edges (sign_consistency >= 0.8 & top100_freq >= 0.6)
        consensus = pd.read_csv(os.path.join(analysis_dir, "consensus_edges_strict.csv"))
        consensus_sources = consensus["source"].tolist()
        self.assertIn("Metabolite_1", consensus_sources)
        self.assertIn("Protein_1", consensus_sources)
        self.assertNotIn("Protein_2", consensus_sources) # 3 pos, 2 neg -> 0.6 < 0.8

if __name__ == '__main__':
    unittest.main()
