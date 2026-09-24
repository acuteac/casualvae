import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from casualvae.module_aclf_partial_multimodal import (
    PartialAnchoredCausalVAE,
    PartialMultiOmicsDataset,
    TrainConfig,
    extract_outputs,
    run_joint_epoch,
    run_pretrain_epoch,
    train_model,
)


def build_model():
    return PartialAnchoredCausalVAE(
        input_dims={"anchor": 1, "y_cont": 1, "y_bin": 1, "protein": 2, "metabolite": 2},
        shared_dims={"protein": 2, "metabolite": 2, "outcome": 1},
        private_dims={"protein": 1, "metabolite": 1},
        hidden_dims={"protein": [4], "metabolite": [4]},
        outcome_exogenous_mode="zero",
    )


def build_batch():
    return {
        "u": torch.zeros(4, 1),
        "y_cont": torch.linspace(-1.0, 1.0, 4).unsqueeze(1),
        "y_bin": torch.tensor([[0.0], [1.0], [0.0], [1.0]]),
        "x_p": torch.tensor([[0.2, -0.4], [0.5, 0.1], [-0.3, 0.8], [0.7, -0.2]]),
        "m_p": torch.ones(4, 2),
        "x_m": torch.tensor([[-0.1, 0.3], [0.6, -0.5], [0.4, 0.9], [-0.7, 0.2]]),
        "m_m": torch.ones(4, 2),
        "avail_p": torch.ones(4, 1),
        "avail_m": torch.ones(4, 1),
        "frac_p": torch.ones(4, 1),
        "frac_m": torch.ones(4, 1),
    }


class ValidationModeTests(unittest.TestCase):
    def test_joint_validation_uses_eval_and_no_grad(self):
        model = build_model()
        observed = []
        hook = model.scm.register_forward_pre_hook(
            lambda _module, _args: observed.append((model.training, torch.is_grad_enabled()))
        )
        try:
            run_joint_epoch(model, [build_batch()], TrainConfig(device="cpu", joint_warmup_epochs=0), None, 0, {})
        finally:
            hook.remove()

        self.assertEqual(observed, [(False, False)])
        self.assertFalse(model.training)
        self.assertTrue(all(not module.training for module in model.modules()))

    def test_pretrain_validation_uses_eval_and_no_grad(self):
        model = build_model()
        observed = []
        hook = model.enc_p.register_forward_pre_hook(
            lambda _module, _args: observed.append((model.training, torch.is_grad_enabled()))
        )
        try:
            run_pretrain_epoch(model, [build_batch()], "protein", TrainConfig(device="cpu"), None)
        finally:
            hook.remove()

        self.assertEqual(observed, [(False, False)])
        self.assertFalse(model.training)

    def test_training_keeps_train_mode_and_gradients(self):
        model = build_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        observed = []
        hook = model.scm.register_forward_pre_hook(
            lambda _module, _args: observed.append((model.training, torch.is_grad_enabled()))
        )
        try:
            run_joint_epoch(model, [build_batch()], TrainConfig(device="cpu", joint_warmup_epochs=0), optimizer, 0, {})
        finally:
            hook.remove()

        self.assertEqual(observed, [(True, True)])
        self.assertTrue(model.training)


class BestCheckpointTauTests(unittest.TestCase):
    def test_train_model_returns_tau_from_best_validation_epoch(self):
        model = build_model()
        cfg = TrainConfig(
            device="cpu",
            pretrain_protein_epochs=0,
            pretrain_metabolite_epochs=0,
            joint_epochs=3,
            early_stopping_patience=10,
        )
        validation_losses = [3.0, 1.0, 2.0]

        def fake_joint_epoch(_model, _loader, _cfg, optimizer, epoch, _weights):
            tau = [2.0, 1.25, 0.5][epoch]
            loss = float(epoch + 10) if optimizer is not None else validation_losses[epoch]
            return {"loss": loss, "tau": tau, "recon": 0.0, "outcome_cont": 0.0,
                    "outcome_bin": 0.0, "dag": 0.0, "use_scm": 1.0}

        with patch("casualvae.module_aclf_partial_multimodal.run_joint_epoch", side_effect=fake_joint_epoch):
            history, best_epoch, best_tau = train_model(
                model, {"paired_train": None, "paired_val": None}, cfg
            )

        self.assertEqual(len(history), 3)
        self.assertEqual(best_epoch, 2)
        self.assertEqual(best_tau, 1.25)

    def test_nonfinite_validation_uses_training_tau_with_training_loss(self):
        model = build_model()
        cfg = TrainConfig(
            device="cpu",
            pretrain_protein_epochs=0,
            pretrain_metabolite_epochs=0,
            joint_epochs=3,
            early_stopping_patience=10,
        )
        training_losses = [3.0, 1.0, 2.0]

        def fake_joint_epoch(_model, _loader, _cfg, optimizer, epoch, _weights):
            tau = [2.0, 1.25, 0.5][epoch] if optimizer is not None else 0.0
            loss = training_losses[epoch] if optimizer is not None else float("inf")
            return {"loss": loss, "tau": tau, "recon": 0.0, "outcome_cont": 0.0,
                    "outcome_bin": 0.0, "dag": 0.0, "use_scm": 1.0}

        with patch("casualvae.module_aclf_partial_multimodal.run_joint_epoch", side_effect=fake_joint_epoch):
            _history, best_epoch, best_tau = train_model(
                model, {"paired_train": None, "paired_val": None}, cfg
            )

        self.assertEqual(best_epoch, 2)
        self.assertEqual(best_tau, 1.25)

    def test_run_summary_records_best_epoch_and_tau(self):
        model = build_model().eval()
        merged = pd.DataFrame({
            "eid": [1, 2, 3, 4], "anchor": [0.0, 0.1, -0.1, 0.2],
            "y_cont": [0.2, -0.3, 0.5, 0.1], "y_bin": [0.0, 1.0, 0.0, 1.0],
            "p1": [0.1, 0.2, 0.3, 0.4], "p2": [-0.1, -0.2, -0.3, -0.4],
            "m1": [0.5, 0.6, 0.7, 0.8], "m2": [-0.5, -0.6, -0.7, -0.8],
        })
        dataset = PartialMultiOmicsDataset(
            merged, ["anchor"], ["y_cont"], ["y_bin"], ["p1", "p2"], ["m1", "m2"]
        )

        with tempfile.TemporaryDirectory() as tmp:
            extract_outputs(
                model, dataset, DataLoader(dataset, batch_size=2), tmp,
                tau=1.25, best_epoch=2, best_tau=1.25,
            )
            summary = json.loads((Path(tmp) / "run_summary.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["best_epoch"], 2)
        self.assertEqual(summary["best_tau"], 1.25)


if __name__ == "__main__":
    unittest.main()
