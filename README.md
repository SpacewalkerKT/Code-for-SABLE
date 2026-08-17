# SABLE: A Sparse Bayesian Meta-Learning Framework for Few-Shot Traffic Incident Recognition

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository provides the revised SABLE implementation, documented experiment configurations, and a lightweight execution workflow for method inspection and controlled re-execution of the study submitted to *IEEE Open Journal of Intelligent Transportation Systems*.

## Overview

SABLE is a few-shot traffic incident recognition framework for single-channel traffic-volume sequences. The released configuration uses volume-only inputs with sequence length `L=25` and constructs episodic tasks from `DESCRIPTION + Week + tag` labels, where the dataset field `tag` denotes the peak-period indicator (`PeakPeriod`). The task split is task-level with DESCRIPTION grouping and does not hold out complete DESCRIPTION categories as separate splits.

The framework combines:

| Component | Role |
|---|---|
| B-SMALL Transformer base learner | Encodes 25-step volume sequences for episodic classification. |
| MAML optimization | Performs task-specific adaptation through inner-loop and outer-loop updates. |
| SparseVDLinear layers | Provide sparse Bayesian regularization in major linear transformations. |
| Distance-based hard-negative selection | Ranks candidate negatives using positive-support standardization and Euclidean distance. |
| Focal/class-weighted loss | Emphasizes hard examples and positive-class errors. |
| Validation-only threshold calibration | Selects a global threshold on validation records and applies it without test-time retuning. |

## Repository Structure

```text
.
├── main.py                         # Command-line entry point
├── run.sh                          # Code Ocean / shell convenience runner
├── requirements.txt                # Python dependencies
├── src/                            # SABLE implementation and baseline models
├── experiments/configs/            # Released experiment configurations
├── data/                           # Data placement notes
└── LICENSE                         # Code license
```

## Environment

The code is intended for Python 3.9 or later with PyTorch 2.x. Install dependencies with:

```bash
pip install -r requirements.txt
```

GPU acceleration is recommended for full training, but the quick-test configuration is intended to validate the environment and execution path with a much smaller budget.

## Data

The documented workflow uses the processed CSV files in `data/`. Their source, expected location, and dataset citation are summarized in [`data/README.md`](data/README.md).

The associated dataset record is available through IEEE DataPort (DOI: 10.21227/d6ht-c375).

## Quick Start

Use the lightweight configuration for an initial environment and pipeline check:

```bash
python main.py --config experiments/configs/ablation/sable_quick_test.yaml --device cpu
```

or through the shell runner:

```bash
CONFIG=experiments/configs/ablation/sable_quick_test.yaml ./run.sh --device cpu
```

The quick-test run is not a new experiment for the paper and should not be used as a replacement for archived numerical results.

## Main Configuration

Paper-oriented settings are defined in the released YAML configuration files. Please refer to the configuration files for the exact training and evaluation budgets used by each experiment.

The main SABLE configuration is:

```bash
python main.py --config experiments/configs/ablation/sable_full.yaml
```

Key verified settings in the released configuration include:

| Item | Setting |
|---|---|
| Input channel | Volume only |
| Sequence length | `L=25` |
| Task construction | `DESCRIPTION + Week + tag` (`tag` = `PeakPeriod`) |
| Split protocol | Task-level, DESCRIPTION-grouped split |
| Support set | 5 target positives + 5 selected negatives |
| Query set | 10 target positives + 10 selected negatives |
| Negative pool | `K+Q=15` negatives |
| Other-group anomaly ratio | `rho_other=0.5` |
| Negative composition | 7 other-group anomaly samples + 8 normal samples |
| HNM oversampling factor | 5 |
| HNM ranking | Positive-support standardization + Euclidean distance |
| Optimizer | Adam |
| Inner / outer learning rates | 0.001 / 0.001 |
| Focal-loss gamma | 2.0 |
| Positive-class weight | 3.0 |
| Thresholding | Validation-only global threshold; no test-time retuning |

## Outputs

Runtime outputs may include logs, checkpoints, metrics, task-sampling records, and figure-support CSV files. Generated outputs should be kept separate from archived result files. The default code path may create directories such as `cache/`, `outputs/`, `metrics/`, `results/`, or `src/checkpoints/` depending on the configuration.

## Reproducibility Boundary

This repository supports method inspection and controlled re-execution under the documented configuration. Quick-test outputs are intended for environment and pipeline validation.

## Citation

Please cite the corresponding manuscript and this software release metadata when using the code. See `CITATION.cff` for machine-readable citation information.

## License

The code is released under the MIT License. See [LICENSE](LICENSE).

The software license applies to the source code. Data use remains subject to the cited data sources and dataset record.
