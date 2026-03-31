# SABLE: A Sparse Bayesian Meta-Learning Framework for Imbalanced Few-Shot Traffic Anomaly Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![Dataset: IEEE DataPort](https://img.shields.io/badge/Dataset-IEEE_DataPort-blue.svg)](#data)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Official PyTorch implementation of the paper **"SABLE: A Sparse Bayesian Meta-Learning Framework for Imbalanced Few-Shot Traffic Anomaly Detection"** (Submitted to *IEEE Open Journal of Intelligent Transportation Systems*).

## Overview

SABLE is a few-shot anomaly detection framework tailored for traffic sensor time series. It addresses the critical challenge of detecting rare, newly emerging traffic anomalies (accidents/incidents) under extreme class imbalance (Imbalance Ratio > 5000) with highly limited labeled examples. 

Given feature sequences collected from multiple sensors, the framework predicts accident labels while prioritizing the reduction of costly false negatives. The framework integrates the following key components:

| Component | Description |
|-----------|-------------|
| **Base Learner (BNNs)** | Bayesian Neural Networks within a MAML framework for robust uncertainty estimation. |
| **Sparse VD** | Sparse Variational Dropout (Regularization) to encourage weight sparsity and reduce meta-overfitting. |
| **Imbalance-Aware Module** | Addresses data imbalance dynamically at both the episodic task level and the loss level. |
| **Hard Negative Mining** | Strategic selection of challenging normal samples to refine decision boundaries. |
| **Constrained Risk Calibration** | Adaptive threshold selection to optimize recall while maintaining precision constraints. |

---

## Quick Reproduction

To reproduce the main results from the paper, run:

```bash
# Clone the repository
git clone [https://github.com/SpacewalkerKT/SABLE-Traffic-Anomaly.git](https://github.com/SpacewalkerKT/SABLE-Traffic-Anomaly.git)
cd SABLE-Traffic-Anomaly

# Install dependencies
pip install -r requirements.txt

# Run full SABLE model (500 episodes, recommended)
python main.py --config experiments/configs/ablation/sable_full.yaml --episodes 500

# Results will be saved to the outputs/ directory
````

-----

## Environment Setup

### Requirements

  - Python 3.9 or higher
  - PyTorch 2.0 or higher
  - CUDA 11.0+ (highly recommended for GPU acceleration)

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

-----

## Data

The benchmark dataset for traffic anomaly detection was constructed utilizing the California Performance Measurement System (PeMS) from District 3 (2023). It encompasses 841 detectors across 20 freeways, featuring **118 fine-grained anomaly families** with an extreme imbalance ratio (IR) of 5201.87.

**Note:** The raw traffic sensor data used in this study are highly aggregated at the macro level and **do not contain any personally identifiable information (PII)** or individual vehicle trajectories.

### Accessing the Dataset

The complete processed dataset is publicly hosted on **IEEE DataPort**: `[Insert your IEEE DataPort Link Here]`

1.  Download `incident_data.csv` and `common_data.csv` from the link above.
2.  Place the downloaded files into the `data/` directory.

### Data Directory Structure

```text
data/
    ├── incident_data.csv  # 118 anomaly families
    └── common_data.csv    # Normal traffic data
```

-----

## Quick Start

### Reproduce Paper Results

#### Ablation Study (Main Results)

```bash
# Run full SABLE (baseline with all components)
python main.py --config experiments/configs/ablation/sable_full.yaml --episodes 500

# Run without Sparse VD
python main.py --config experiments/configs/ablation/sable_no_sparsevd.yaml

# Run without Imbalance-Aware Components
python main.py --config experiments/configs/ablation/sable_no_classweight.yaml

# Run without Hard Negative Mining
python main.py --config experiments/configs/ablation/sable_no_hardneg.yaml

# Run without Constrained Risk Calibration (Thresholding)
python main.py --config experiments/configs/ablation/sable_no_threshold.yaml
```

#### Comparison Experiments (Baselines)

```bash
# MAML baseline
python main.py --config experiments/configs/comparison/compare_maml.yaml

# Prototypical Networks
python main.py --config experiments/configs/comparison/compare_protonet.yaml

# TSA-MAML
python main.py --config experiments/configs/comparison/compare_tsa_maml.yaml
```

#### K-Shot Analysis

```bash
# 1-shot learning
python main.py --config experiments/configs/kshot/sable_k1.yaml

# 5-shot learning (Default setting)
python main.py --config experiments/configs/kshot/sable_k5.yaml
```

-----

## Configuration

Experiments are configured via YAML files. Key hyperparameters include:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `K_SHOT` | 5 | Support set size (number of examples per class) |
| `Q_SHOT` | 10 | Query set size |
| `EPISODES` | 500 | Number of training episodes |
| `lr_inner` | 0.001 | Inner-loop learning rate |
| `lr_meta` | 0.0001 | Meta-learning rate |
| `SEED` | 42 | Random seed for reproducibility (used for family-wise splitting) |

-----

## Results & Outputs

The framework tracks robust metrics suitable for extreme imbalance, primarily focusing on **PR-AUC, ROC-AUC, F1-Score, and Recall**. Accuracy (ACC) is logged but not relied upon as a primary metric due to the IR.

Upon execution, results are saved in the `results/` directory:

  - `sable_pr_curve.csv`: Precision-Recall curve data.
  - `kshot_auc_comparison.csv`: Performance across varying *K* shots.
  - `sable_family_auc.csv`: Family-level cross-class generalization performance.

-----

## Known Limitations & Future Work

1.  **Single-Region Scope**: Current experiments are primarily conducted on single-region PeMS data (District 3) utilizing volume time series. Future work will consider cross-region and cross-scenario transfer.
2.  **Computational Overhead**: While constrained threshold calibration improves risk control during deployment, it introduces additional validation and inference overhead compared to lighter architectures.
3.  **Reproducibility Variance**: While `seed=42` is fixed, slight metric variations (±0.5%) are possible across different hardware (CPU vs. GPU) or CUDA versions.

-----

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

## Contact

For questions, discussions, or issues:

  - Open an issue on this repository
  - Email: numknw@163.com

## Acknowledgments

This research was supported by the Intelligent Policing Key Laboratory of Sichuan Province. We also extend our gratitude to the anonymous reviewers for their valuable feedback.