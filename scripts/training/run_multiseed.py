import os
import sys
import argparse
import subprocess
import traceback
from pathlib import Path

# 修复 Anaconda/Windows 环境下常见的 OpenMP 冲突报错
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_SCRIPT = PROJECT_ROOT / "src" / "casualvae" / "module_aclf_partial_multimodal.py"
DEFAULT_BASE_OUT_DIR = PROJECT_ROOT / "artifacts" / "multiseed" / "fixed_split"
STRUCTURED_DATA_DIR = PROJECT_ROOT / "data" / "structured_full"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "aclf_full_multimodal_config.json"

def parse_args():
    parser = argparse.ArgumentParser(description="Run Multiseed Training for CasualVAE")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training (cuda/cpu)")
    parser.add_argument("--split-seed", type=int, default=42, help="Fixed train/validation split seed")
    parser.add_argument(
        "--model-seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789, 2026],
        help="Model/training randomness seeds",
    )
    parser.add_argument(
        "--base-out-dir",
        default=str(DEFAULT_BASE_OUT_DIR),
        help="Directory containing one model_seed_<seed> directory per run",
    )
    return parser.parse_args()


def model_seed_output_dir(base_out_dir, model_seed):
    return os.path.join(base_out_dir, f"model_seed_{model_seed}")


def build_training_command(
    python_executable,
    model_seed,
    split_seed,
    out_dir,
    device,
    inputs,
):
    return [
        python_executable,
        str(MODEL_SCRIPT),
        "--anchors_csv", inputs["anchors_csv"],
        "--phenotype_csv", inputs["phenotype_csv"],
        "--protein_csv", inputs["protein_csv"],
        "--metabolite_csv", inputs["metabolite_csv"],
        "--config_json", inputs["config_json"],
        "--out_dir", out_dir,
        "--split_seed", str(split_seed),
        "--model_seed", str(model_seed),
        "--device", device,
    ]

def main():
    args = parse_args()

    seeds = args.model_seeds
    base_out_dir = args.base_out_dir
    os.makedirs(base_out_dir, exist_ok=True)

    inputs = {
        "anchors_csv": str(STRUCTURED_DATA_DIR / "anchors_structured.csv"),
        "phenotype_csv": str(STRUCTURED_DATA_DIR / "phenotype_structured.csv"),
        "protein_csv": str(STRUCTURED_DATA_DIR / "protein_full_structured.csv"),
        "metabolite_csv": str(STRUCTURED_DATA_DIR / "metabolite_full_structured.csv"),
        "config_json": str(DEFAULT_CONFIG),
    }

    # check inputs exist
    for f in inputs.values():
        if not os.path.exists(f):
            print(f"Warning: Expected input file {f} does not exist.")
            print("Make sure you have run data preparation steps before running this script.")
            return

    for seed in seeds:
        print(f"\n=======================================================")
        print(f"Starting training for seed {seed}")
        print(f"=======================================================\n")

        seed_out_dir = model_seed_output_dir(base_out_dir, seed)
        os.makedirs(seed_out_dir, exist_ok=True)

        cmd = build_training_command(
            python_executable=sys.executable,
            model_seed=seed,
            split_seed=args.split_seed,
            out_dir=seed_out_dir,
            device=args.device,
            inputs=inputs,
        )

        print(f"Running command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
            print(f"\n✅ Seed {seed} training completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error: Seed {seed} training failed with exit code {e.returncode}.")
            traceback.print_exc()
        except Exception as e:
            print(f"\n❌ Error: Seed {seed} training encountered an unexpected error.")
            traceback.print_exc()

    print("\nAll multiseed runs finished.")

if __name__ == "__main__":
    main()
