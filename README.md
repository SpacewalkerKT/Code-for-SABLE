# SABLE: Sparse Adaptive Balanced Learning for Few-Shot Traffic Incident Classification

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Quick Reproduction

To reproduce the main results from the paper, run:

```bash
# Install dependencies
pip install -r requirements.txt

# Run full SABLE model (500 episodes, recommended)
python main.py --config experiments/configs/ablation/sable_full.yaml --episodes 500

# Results will be saved to outputs/ directory
```

For detailed experiment configurations, see the sections below.

---

## Overview

SABLE is a meta-learning framework designed for few-shot traffic incident classification. It addresses the challenge of classifying new types of traffic incidents with limited labeled examples (typically 5-10 samples) by integrating six key components:

| Component | Description |
|-----------|-------------|
| **MAML** | Model-Agnostic Meta-Learning for fast adaptation to new incident types |
| **Sparse VD** | Sparse Variational Dropout for model regularization and sparsity |
| **Class Weighting** | Adaptive class balancing for severe imbalance (up to 50:1 ratio) |
| **Hard Negative Mining** | Strategic selection of challenging negative samples |
| **Adaptive Threshold** | Dynamic threshold optimization for precision-recall balance |
| **Recall Priority** | Fβ optimization prioritizing recall for safety-critical scenarios |

## Environment Setup

### Requirements

- Python 3.9 or higher
- PyTorch 2.0 or higher
- CUDA 11.0+ (recommended for GPU acceleration)

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python main.py --help
```

### Quick Test

```bash
# Test installation with a quick run (10 episodes)
python main.py --config experiments/configs/ablation/sable_full.yaml --episodes 10
```

## Quick Start

### Reproduce Paper Results

#### Ablation Study (Main Results)

```bash
# Run full SABLE (baseline with all components)
python main.py --config experiments/configs/ablation/sable_full.yaml --episodes 500

# Run without Sparse VD
python main.py --config experiments/configs/ablation/sable_no_sparsevd.yaml

# Run without Class Weighting
python main.py --config experiments/configs/ablation/sable_no_classweight.yaml

# Run without Hard Negative Mining
python main.py --config experiments/configs/ablation/sable_no_hardneg.yaml

# Run without Adaptive Threshold
python main.py --config experiments/configs/ablation/sable_no_threshold.yaml

# Run without Meta-Learning
python main.py --config experiments/configs/ablation/sable_no_metalearning.yaml

# Run without Recall Priority
python main.py --config experiments/configs/ablation/sable_no_recallpriority.yaml
```

#### Comparison Experiments

```bash
# MAML baseline
python main.py --config experiments/configs/comparison/compare_maml.yaml

# Prototypical Networks
python main.py --config experiments/configs/comparison/compare_protonet.yaml

# Logistic Regression
python main.py --config experiments/configs/comparison/compare_logreg.yaml

# Joint Training
python main.py --config experiments/configs/comparison/compare_joint_training.yaml

# TSA-MAML
python main.py --config experiments/configs/comparison/compare_tsa_maml.yaml
```

#### K-Shot Analysis

```bash
# 1-shot learning
python main.py --config experiments/configs/kshot/sable_k1.yaml

# 3-shot learning
python main.py --config experiments/configs/kshot/sable_k3.yaml

# 5-shot learning (default)
python main.py --config experiments/configs/kshot/sable_k5.yaml

# 10-shot learning
python main.py --config experiments/configs/kshot/sable_k10.yaml
```

### Custom Experiments

```bash
# Train with custom parameters
python main.py --config experiments/configs/ablation/sable_full.yaml \
    --episodes 1000 \
    --device cuda:0 \
    --seed 42

# Run on CPU (slower but works without GPU)
python main.py --config experiments/configs/ablation/sable_full.yaml --device cpu
```

## Data

### Data Directory Structure

The project expects the following data files in `data/raw/`:
- `cleaned_incident_data.csv` (130MB) - Traffic incident records
- `cleaned_common_data.csv` (280MB) - Normal traffic data

**Data paths can be customized via YAML config:**
```yaml
data:
  raw_dir: "data/raw"
  incident_file: "cleaned_incident_data.csv"
  normal_file: "cleaned_common_data.csv"
```

### Included Data

This repository includes the complete dataset used in the paper experiments:
- **Incident data**: 583,753 records from traffic incidents
- **Normal data**: 1,161,709 records from normal traffic patterns
- **Features**: 125-dimensional feature vectors per sample

### Dataset Format

### Expected Columns

| Column | Description |
|--------|-------------|
| `IncidentId` | Unique incident identifier |
| `SensorNumber` | Sensor/station ID |
| `Week` | Week number (1-52) |
| `label` | Class label (0=normal, 1=incident) |
| `feat_1` to `feat_125` | Extracted features (125 dimensions) |

### Sample Data

For testing purposes, you can create synthetic data following the expected format:

```python
import pandas as pd
import numpy as np

# Create sample incident data
incident_data = pd.DataFrame({
    'IncidentId': range(100),
    'SensorNumber': np.random.randint(1, 50, 100),
    'Week': np.random.randint(1, 53, 100),
    'label': np.ones(100),
    **{f'feat_{i}': np.random.randn(100) for i in range(1, 126)}
})

# Create sample normal data
normal_data = pd.DataFrame({
    'IncidentId': range(100, 1000),
    'SensorNumber': np.random.randint(1, 50, 900),
    'Week': np.random.randint(1, 53, 900),
    'label': np.zeros(900),
    **{f'feat_{i}': np.random.randn(900) for i in range(1, 126)}
})

incident_data.to_csv('data/raw/cleaned_incident_data.csv', index=False)
normal_data.to_csv('data/raw/cleaned_common_data.csv', index=False)
```

## Configuration

### Configuration File Structure

Experiments are configured via YAML files:

```yaml
# Example configuration structure
experiment:
  name: sable_full_pilot
  description: Full SABLE with all components

training:
  episodes: 500
  k_shot: 5
  q_shot: 10
  lr_inner: 0.001
  lr_meta: 0.0001
  inner_steps: 5
  seed: 42

model:
  # For comparison experiments only
  name: MAML  # Options: MAML, ProtoNet, LogReg, JointTraining, TSA_MAML

components:
  use_sparse_vd: true
  use_class_weight: true
  use_hard_negatives: true
  use_threshold_search: true
  use_recall_priority: true
  meta_learning: true

  # Class weighting parameters
  use_fixed_pos_weight: true
  fixed_pos_weight: 1.6

  # Sparse VD parameters
  kl_scale: 0.01

recall_priority:
  r_start: 0.60
  r_target: 0.95
  r_warmup_frac: 0.5
  rmin_schedule: linear
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `K_SHOT` | 5 | Support set size (number of examples per class) |
| `Q_SHOT` | 10 | Query set size |
| `EPISODES` | 500 | Number of training episodes |
| `lr_inner` | 0.001 | Inner-loop learning rate |
| `lr_meta` | 0.0001 | Meta-learning rate |
| `inner_steps` | 5 | Gradient steps in inner loop |
| `SEED` | 42 | Random seed for reproducibility |

## Results

### Outputs Directory

This repository includes paper experiment results in `results/`:
- **fig2** - PR curve data (Fig 2 in paper)
  - `sable_pr_curve.csv` - SABLE PR curve data
  - `tsa_maml_pr_curve.csv` - TSA-MAML PR curve data
- **fig3** - K-shot AUC comparison (Fig 3 in paper)
  - `kshot_auc_comparison.csv` - K-shot AUC results
- **fig4** - Family-level AUC analysis (Fig 4 in paper)
  - `sable_family_auc.csv` - SABLE family-level AUC
  - `tsa_maml_family_auc.csv` - TSA-MAML family-level AUC
- **figures** - PDF and PNG chart files
  - `fig2_pr_curve.{pdf,png}` - Precision-Recall curves
  - `fig3_kshot_auc.{pdf,png}` - K-shot AUC comparison
  - `fig4_family_auc_boxplot.{pdf,png}` - Family-level AUC boxplot
- **comparison_summary.csv** - Baseline comparison results (Table II in paper)
- **ablation_summary.csv** - Ablation study results (Table III in paper)

### Key Metrics

The framework tracks the following metrics:

| Metric | Description |
|--------|-------------|
| ACC | Accuracy |
| F1 | F1 Score |
| REC | Recall (Sensitivity) |
| PREC | Precision |
| AUC | Area Under ROC Curve |
| AUPRC | Area Under Precision-Recall Curve |

### Reproducibility

- Random seed is set to 42 by default
- Results may vary slightly due to:
  - Different hardware (CPU vs GPU)
  - PyTorch version differences
  - CUDA version differences

Expected variance: ±0.5% on main metrics

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Solution: Use CPU mode
python main.py --config configs/default.yaml --device cpu
```

**2. Data File Not Found**
```bash
# Solution: Ensure data is in data/raw/ directory
# Check paths in your config file
```

**3. Import Errors**
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**4. Module Not Found**
```bash
# Solution: Ensure you're in the project root and venv is activated
cd SABLE
source venv/bin/activate
python main.py --config experiments/configs/...
```

## Known Limitations

1. **Data Privacy**: Full dataset cannot be shared due to privacy restrictions
2. **Hardware**: GPU recommended but not required (CPU mode available)
3. **Reproducibility**: Slight variations possible across different hardware
4. **Sample Data**: Provided sample data is for testing only

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions and issues:
- Open an issue on GitHub
- Email: numknw@163.com

## Acknowledgments

This research was supported by Intelligent Policing Key Laboratory of Sichuan Province. We thank the anonymous reviewers for valuable feedback.
