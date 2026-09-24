import os
from pathlib import Path
import unittest

from scripts.training import run_multiseed


class FixedSplitRunnerTests(unittest.TestCase):
    def test_default_paths_use_canonical_project_layout(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(Path(run_multiseed.MODEL_SCRIPT), root / "src" / "casualvae" / "module_aclf_partial_multimodal.py")
        self.assertEqual(Path(run_multiseed.DEFAULT_BASE_OUT_DIR), root / "artifacts" / "multiseed" / "fixed_split")

    def test_build_training_command_passes_independent_seeds(self):
        command = run_multiseed.build_training_command(
            python_executable="python.exe",
            model_seed=123,
            split_seed=42,
            out_dir="artifacts/multiseed/fixed_split/model_seed_123",
            device="cpu",
            inputs={
                "anchors_csv": "anchors.csv",
                "phenotype_csv": "phenotype.csv",
                "protein_csv": "protein.csv",
                "metabolite_csv": "metabolite.csv",
                "config_json": "config.json",
            },
        )

        self.assertIn("--split_seed", command)
        self.assertEqual(command[command.index("--split_seed") + 1], "42")
        self.assertIn("--model_seed", command)
        self.assertEqual(command[command.index("--model_seed") + 1], "123")
        self.assertNotIn("--seed", command)

    def test_default_output_directory_names_model_seed(self):
        actual = run_multiseed.model_seed_output_dir("artifacts/multiseed/fixed_split", 2026)
        expected = os.path.join("artifacts", "multiseed", "fixed_split", "model_seed_2026")
        self.assertEqual(os.path.normpath(actual), os.path.normpath(expected))


if __name__ == "__main__":
    unittest.main()
