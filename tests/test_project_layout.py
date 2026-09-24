from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProjectLayoutTests(unittest.TestCase):
    def test_canonical_project_paths_exist(self):
        expected = [
            ROOT / "src" / "casualvae" / "module_aclf_partial_multimodal.py",
            ROOT / "scripts" / "training" / "run_multiseed.py",
            ROOT / "scripts" / "analysis" / "analyze_multiseed.py",
            ROOT / "configs" / "aclf_full_multimodal_config.json",
            ROOT / "data" / "structured_full",
            ROOT / "artifacts" / "multiseed" / "fixed_split",
            ROOT / "artifacts" / "analysis" / "multiseed" / "stability_comparison.csv",
        ]
        missing = [str(path) for path in expected if not path.exists()]
        self.assertEqual(missing, [])

    def test_obsolete_root_locations_are_absent(self):
        obsolete = [
            ROOT / "module_aclf_partial_multimodal.py",
            ROOT / "run_multiseed.py",
            ROOT / "analyze_multiseed.py",
            ROOT / "outputs",
        ]
        remaining = [str(path) for path in obsolete if path.exists()]
        self.assertEqual(remaining, [])


if __name__ == "__main__":
    unittest.main()
