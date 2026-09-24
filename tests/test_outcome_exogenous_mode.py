import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from casualvae.module_aclf_partial_multimodal import (
    PartialAnchoredCausalVAE,
    PartialMultiOmicsDataset,
    extract_outputs,
)


def build_model(mode="zero", sigma=1.0, shared_dims=None):
    return PartialAnchoredCausalVAE(
        input_dims={"anchor": 1, "y_cont": 1, "y_bin": 1, "protein": 2, "metabolite": 2},
        shared_dims=shared_dims if shared_dims is not None else {"protein": 2, "metabolite": 2, "outcome": 1},
        private_dims={"protein": 1, "metabolite": 1},
        hidden_dims={"protein": [4], "metabolite": [4]},
        outcome_exogenous_mode=mode,
        outcome_exogenous_sigma=sigma,
    )


def build_batch(batch_size=4):
    return {
        "u": torch.zeros(batch_size, 1),
        "y_cont": torch.linspace(-1.0, 1.0, batch_size).unsqueeze(1),
        "y_bin": torch.tensor([[0.0], [1.0], [0.0], [1.0]])[:batch_size],
        "x_p": torch.tensor([[0.2, -0.4], [0.5, 0.1], [-0.3, 0.8], [0.7, -0.2]])[:batch_size],
        "m_p": torch.ones(batch_size, 2),
        "x_m": torch.tensor([[-0.1, 0.3], [0.6, -0.5], [0.4, 0.9], [-0.7, 0.2]])[:batch_size],
        "m_m": torch.ones(batch_size, 2),
        "avail_p": torch.ones(batch_size, 1),
        "avail_m": torch.ones(batch_size, 1),
        "frac_p": torch.ones(batch_size, 1),
        "frac_m": torch.ones(batch_size, 1),
    }


def forward(model, batch, y_cont=None, y_bin=None, use_scm=True):
    return model.forward_joint(
        u=batch["u"],
        y_cont=batch["y_cont"] if y_cont is None else y_cont,
        y_bin=batch["y_bin"] if y_bin is None else y_bin,
        x_p=batch["x_p"],
        m_p=batch["m_p"],
        x_m=batch["x_m"],
        m_m=batch["m_m"],
        avail_p=batch["avail_p"],
        avail_m=batch["avail_m"],
        frac_p=batch["frac_p"],
        frac_m=batch["frac_m"],
        use_scm=use_scm,
        tau=1.0,
    )


def grad_total(parameters):
    return sum(float(p.grad.abs().sum()) for p in parameters if p.grad is not None)


class OutcomeExogenousModeTests(unittest.TestCase):
    def test_encoded_y_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outcome_exogenous_mode"):
            build_model("encoded_y")

    def test_outcome_block_is_required(self):
        for dims in ({"protein": 2, "metabolite": 2},
                     {"protein": 2, "metabolite": 2, "outcome": 0}):
            with self.subTest(dims=dims):
                with self.assertRaisesRegex(ValueError, "outcome"):
                    build_model(shared_dims=dims)

    def test_zero_mode_predictions_do_not_depend_on_observed_y(self):
        model = build_model("zero").eval()
        batch = build_batch()

        torch.manual_seed(11)
        first = forward(model, batch)
        torch.manual_seed(11)
        second = forward(model, batch, y_cont=batch["y_cont"] + 100.0, y_bin=1.0 - batch["y_bin"])

        self.assertNotIn("enc_outcome", first)
        self.assertEqual(first["z_all"].shape[1], 5)
        torch.testing.assert_close(first["z_outcome"], second["z_outcome"])
        torch.testing.assert_close(first["y_cont_hat"], second["y_cont_hat"])
        torch.testing.assert_close(first["y_bin_logits"], second["y_bin_logits"])

    def test_zero_mode_uses_target_source_scm_solve(self):
        model = build_model("zero").eval()
        batch = build_batch()

        torch.manual_seed(19)
        out = forward(model, batch)

        eps_shared = torch.cat(
            [
                out["enc_p"]["shared_eps"] * batch["avail_p"],
                out["enc_m"]["shared_eps"] * batch["avail_m"],
                torch.zeros(4, 1),
            ],
            dim=1,
        )
        rhs = eps_shared + model.anchor_to_context(batch["u"])
        expected = torch.linalg.solve(model.scm.eye - out["a"], rhs.T).T
        torch.testing.assert_close(out["z_all"], expected)

    def test_gaussian_mode_is_independent_of_observed_y(self):
        model = build_model("gaussian", sigma=0.5).eval()
        batch = build_batch()

        torch.manual_seed(23)
        first = forward(model, batch, use_scm=False)
        torch.manual_seed(23)
        second = forward(model, batch, y_cont=batch["y_cont"] - 100.0, use_scm=False)

        self.assertNotIn("enc_outcome", first)
        torch.testing.assert_close(first["z_outcome"], second["z_outcome"])
        self.assertGreater(float(first["z_outcome"].detach().abs().sum()), 0.0)

    def test_outcome_loss_updates_decoders_scm_edges_and_omics_encoders(self):
        model = build_model("zero").eval()
        batch = build_batch()
        out = forward(model, batch)

        loss = F.mse_loss(out["y_cont_hat"], batch["y_cont"]) + F.binary_cross_entropy_with_logits(
            out["y_bin_logits"], batch["y_bin"]
        )
        loss.backward()

        slices = model.scm.block_slices
        outcome_rows = slices["outcome"]
        protein_cols = slices["protein"]
        metabolite_cols = slices["metabolite"]
        self.assertGreater(grad_total(model.dec_outcome.parameters()), 0.0)
        self.assertGreater(float(model.scm.weights.grad[outcome_rows, protein_cols].abs().sum()), 0.0)
        self.assertGreater(float(model.scm.weights.grad[outcome_rows, metabolite_cols].abs().sum()), 0.0)
        self.assertGreater(grad_total(model.enc_p.parameters()), 0.0)
        self.assertGreater(grad_total(model.enc_m.parameters()), 0.0)
        self.assertFalse(hasattr(model, "enc_outcome"))
        self.assertFalse(hasattr(model, "outcome_head"))

    def test_scm_mask_keeps_outcome_as_downstream_block(self):
        model = build_model("zero")
        slices = model.scm.block_slices
        p, m, o = slices["protein"], slices["metabolite"], slices["outcome"]

        self.assertTrue(torch.all(model.scm.mask[o, p] == 1))
        self.assertTrue(torch.all(model.scm.mask[o, m] == 1))
        self.assertTrue(torch.all(model.scm.mask[p, o] == 0))
        self.assertTrue(torch.all(model.scm.mask[m, o] == 0))

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outcome_exogenous_mode"):
            build_model("observed_y")

    def test_run_summary_records_mode_and_exports_compatible_files(self):
        model = build_model("zero").eval()
        merged = pd.DataFrame(
            {
                "eid": [1, 2, 3, 4],
                "anchor": [0.0, 0.1, -0.1, 0.2],
                "y_cont": [0.2, -0.3, 0.5, 0.1],
                "y_bin": [0.0, 1.0, 0.0, 1.0],
                "p1": [0.1, 0.2, 0.3, 0.4],
                "p2": [-0.1, -0.2, -0.3, -0.4],
                "m1": [0.5, 0.6, 0.7, 0.8],
                "m2": [-0.5, -0.6, -0.7, -0.8],
            }
        )
        dataset = PartialMultiOmicsDataset(
            merged=merged,
            anchor_cols=["anchor"],
            phenotype_cont_cols=["y_cont"],
            phenotype_bin_cols=["y_bin"],
            protein_cols=["p1", "p2"],
            metabolite_cols=["m1", "m2"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            extract_outputs(model, dataset, DataLoader(dataset, batch_size=2), tmp, tau=0.5)
            summary = json.loads((Path(tmp) / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["outcome_exogenous_mode"], "zero")
            for filename in (
                "latent_causal_A.csv",
                "decoder_projection_D.csv",
                "feature_projection_G.csv",
                "G_protein_to_outcome.csv",
                "G_metabolite_to_outcome.csv",
            ):
                self.assertTrue((Path(tmp) / filename).is_file(), filename)

    def test_repository_config_defaults_to_zero_mode(self):
        config = json.loads((ROOT / "configs" / "aclf_full_multimodal_config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["outcome_exogenous_mode"], "zero")


if __name__ == "__main__":
    unittest.main()
