#!/usr/bin/env python3
"""
SABLE: Sparse Adaptive Balanced Learning for Event Classification

Main entry point for training and evaluation of the SABLE framework.
This script provides a unified interface for reproducing paper results.

Usage:
    python main.py --config experiments/configs/ablation/sable_full_pilot.yaml
    python main.py --config experiments/configs/ablation/sable_full_pilot.yaml --episodes 500
    python main.py --config experiments/configs/comparison/compare_maml_pilot.yaml --device cuda:0

Paper:
    SABLE: Sparse Adaptive Balanced Learning for Few-Shot Traffic Incident Classification
    IEEE Open Journal of Intelligent Transportation Systems (OJ-ITS)

Author: SABLE Research Team
License: MIT
"""

import argparse
import sys
import os
from pathlib import Path

# Add src directory to Python path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SABLE: Sparse Adaptive Balanced Learning for Event Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full SABLE (ablation baseline)
  python main.py --config experiments/configs/ablation/sable_full.yaml

  # Run with custom number of episodes
  python main.py --config experiments/configs/ablation/sable_full.yaml --episodes 500

  # Run comparison experiment
  python main.py --config experiments/configs/comparison/compare_maml.yaml

  # Run on specific GPU
  python main.py --config experiments/configs/ablation/sable_full.yaml --device cuda:0

  # Run K-shot experiment
  python main.py --config experiments/configs/kshot/sable_k1.yaml

Available Experiments:
  Ablation Study:
    - sable_full.yaml         (Full SABLE)
    - sable_no_sparsevd.yaml  (Without Sparse VD)
    - sable_no_classweight.yaml (Without Class Weighting)
    - sable_no_hardneg.yaml   (Without Hard Negative Mining)
    - sable_no_threshold.yaml (Without Adaptive Threshold)
    - sable_no_metalearning.yaml (Without Meta-Learning)
    - sable_no_recallpriority.yaml (Without Recall Priority)

  Comparison Methods:
    - compare_maml.yaml       (MAML baseline)
    - compare_protonet.yaml   (ProtoNet baseline)
    - compare_logreg.yaml     (Logistic Regression)
    - compare_joint_training.yaml (Joint Training)
    - compare_tsa_maml.yaml   (TSA-MAML)

  K-Shot Analysis:
    - sable_k1.yaml, sable_k3.yaml, sable_k5.yaml, sable_k10.yaml
        """
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to experiment configuration file (YAML format)"
    )

    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=None,
        help="Override number of training episodes (default: use config value)"
    )

    parser.add_argument(
        "--device", "-d",
        type=str,
        default=None,
        help="Device to run on (e.g., 'cuda:0', 'cuda:1', 'cpu'). Default: auto-detect"
    )

    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: use config value, typically 42)"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Override output directory (default: outputs/)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-essential output"
    )

    return parser.parse_args()


def validate_config(config_path: Path) -> bool:
    """Validate that the configuration file exists and is readable."""
    if not config_path.exists():
        print(f"ERROR: Configuration file not found: {config_path}")
        print(f"Please check the path and try again.")
        return False

    if not config_path.suffix.lower() in ['.yaml', '.yml']:
        print(f"WARNING: Configuration file should be a YAML file (.yaml or .yml)")

    return True


def setup_device(device_arg: str = None) -> str:
    """Setup and validate the computing device."""
    import torch

    if device_arg:
        if device_arg.startswith("cuda"):
            if not torch.cuda.is_available():
                print("WARNING: CUDA requested but not available. Falling back to CPU.")
                return "cpu"
        return device_arg

    # Auto-detect best available device
    if torch.cuda.is_available():
        device = "cuda:0"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[Device] Using GPU: {gpu_name}")
        return device
    else:
        print("[Device] CUDA not available, using CPU")
        return "cpu"


def print_banner():
    """Print the SABLE banner."""
    banner = """

                                                                              
                                   
                            
                                   
                                   
                     
                           
                                                                              
   Sparse Adaptive Balanced Learning for Event Classification                 
                                                                              

"""
    print(banner)


def main():
    """Main entry point for SABLE framework."""
    args = parse_args()

    # Print banner
    if not args.quiet:
        print_banner()

    # Validate configuration file
    config_path = Path(args.config)
    if not validate_config(config_path):
        sys.exit(1)

    # Print configuration info
    print(f"[Config] Loading: {config_path}")

    # Setup device
    device = setup_device(args.device)
    if device != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1] if ":" in device else "0"

    # Build command for SABLE_optimized.py
    cmd_args = ["--config", str(config_path)]

    if args.episodes:
        cmd_args.extend(["--episodes", str(args.episodes)])

    # Override sys.argv for SABLE_optimized.py
    original_argv = sys.argv
    sys.argv = ["SABLE_optimized.py"] + cmd_args

    try:
        if not args.quiet:
            print(f"\n{'='*80}")
            print("[SABLE] Starting experiment...")
            print(f"{'='*80}\n")

        # Import and run SABLE_optimized
        # Note: SABLE_optimized.py handles the main execution in its __main__ block
        import runpy
        runpy.run_path(str(PROJECT_ROOT / "src" / "SABLE_optimized.py"), run_name="__main__")

    except KeyboardInterrupt:
        print("\n[SABLE] Experiment interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n[SABLE] ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        sys.argv = original_argv

    if not args.quiet:
        print(f"\n{'='*80}")
        print("[SABLE] Experiment completed successfully")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
