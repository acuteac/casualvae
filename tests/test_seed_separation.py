import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from casualvae.module_aclf_partial_multimodal import (
    PartialAnchoredCausalVAE,
    PartialMultiOmicsDataset,
    TrainConfig,
    extract_outputs,
    make_dataloaders,
)


def build_dataset(n_samples=40):
    merged = pd.DataFrame(
        {
            "eid": list(range(1000, 1000 + n_samples)),
            "anchor": [float(i) for i in range(n_samples)],
            "y": [float(i % 5) for i in range(n_samples)],
            "p1": [float(i) for i in range(n_samples)],
            "p2": [float(i + 1) for i in range(n_samples)],
            "m1": [float(i + 2) for i in range(n_samples)],
            "m2": [float(i + 3) for i in range(n_samples)],
        }
    )
    return PartialMultiOmicsDataset(
        merged,
        anchor_cols=["anchor"],
        phenotype_cont_cols=["y"],
        phenotype_bin_cols=[],
        protein_cols=["p1", "p2"],
        metabolite_cols=["m1", "m2"],
    )


def split_ids(meta):
    return {
        key: tuple(meta[key])
        for key in (
            "protein_train_eids",
            "protein_val_eids",
            "metabolite_train_eids",
            "metabolite_val_eids",
            "paired_train_eids",
            "paired_val_eids",
        )
    }


class SeedSeparationTests(unittest.TestCase):
    def test_same_split_seed_gives_identical_patient_splits_for_all_model_seeds(self):
        dataset = build_dataset()
        cfg = TrainConfig(batch_size=8, val_fraction=0.25, device="cpu")

        _loaders_a, meta_a = make_dataloaders(dataset, cfg, split_seed=42, model_seed=42)
        _loaders_b, meta_b = make_dataloaders(dataset, cfg, split_seed=42, model_seed=2026)

        self.assertEqual(split_ids(meta_a), split_ids(meta_b))
        self.assertEqual(meta_a["split_seed"], 42)
        self.assertEqual(meta_b["model_seed"], 2026)

    def test_changing_model_seed_changes_shuffle_stream_not_patient_split(self):
        dataset = build_dataset()
        cfg = TrainConfig(batch_size=8, val_fraction=0.25, device="cpu")

        loaders_a, meta_a = make_dataloaders(dataset, cfg, split_seed=42, model_seed=42)
        loaders_b, meta_b = make_dataloaders(dataset, cfg, split_seed=42, model_seed=123)

        self.assertEqual(split_ids(meta_a), split_ids(meta_b))
        order_a = [eid for batch in loaders_a["paired_train"] for eid in batch["eid"].tolist()]
        order_b = [eid for batch in loaders_b["paired_train"] for eid in batch["eid"].tolist()]
        self.assertNotEqual(order_a, order_b)

    def test_changing_split_seed_changes_patient_split(self):
        dataset = build_dataset()
        cfg = TrainConfig(batch_size=8, val_fraction=0.25, device="cpu")

        _loaders_a, meta_a = make_dataloaders(dataset, cfg, split_seed=42, model_seed=42)
        _loaders_b, meta_b = make_dataloaders(dataset, cfg, split_seed=43, model_seed=42)

        self.assertNotEqual(split_ids(meta_a), split_ids(meta_b))

    def test_run_summary_records_split_and_model_seed_without_changing_graph_outputs(self):
        dataset = build_dataset(n_samples=4)
        model = PartialAnchoredCausalVAE(
            input_dims={"anchor": 1, "y_cont": 1, "y_bin": 0, "protein": 2, "metabolite": 2},
            shared_dims={"protein": 2, "metabolite": 2, "outcome": 1},
            private_dims={"protein": 1, "metabolite": 1},
            hidden_dims={"protein": [4], "metabolite": [4]},
            outcome_exogenous_mode="zero",
        ).eval()

        with tempfile.TemporaryDirectory() as tmp:
            extract_outputs(
                model,
                dataset,
                DataLoader(dataset, batch_size=2),
                tmp,
                tau=1.0,
                split_seed=42,
                model_seed=123,
            )
            summary = json.loads((Path(tmp) / "run_summary.json").read_text(encoding="utf-8"))
            expected_graph_files = {
                "G_protein_to_outcome.csv",
                "G_metabolite_to_outcome.csv",
                "decoder_projection_D.csv",
            }
            actual_files = {path.name for path in Path(tmp).iterdir()}

        self.assertEqual(summary["split_seed"], 42)
        self.assertEqual(summary["model_seed"], 123)
        self.assertTrue(expected_graph_files.issubset(actual_files))


if __name__ == "__main__":
    unittest.main()
