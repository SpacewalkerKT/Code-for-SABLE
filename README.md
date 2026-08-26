# SABLE: A Sparse Bayesian Meta-Learning Framework for Few-Shot Traffic Incident Recognition

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository provides the implementation of **SABLE**, a sparse Bayesian meta-learning framework for Few-Shot traffic incident recognition.

The repository accompanies our manuscript submitted to *IEEE Open Journal of Intelligent Transportation Systems* and provides the source code, documented experiment configurations, and a lightweight workflow for method inspection and controlled re-execution.

## Overview

The released repository is intended to support inspection of the proposed method and execution under the documented configurations. Detailed methodological definitions, experimental protocols, and result analysis are provided in the associated manuscript.

## Repository Structure

```text
.
├── main.py                    # Command-line entry point
├── run.sh                     # Shell
├── requirements.txt           # Python dependencies
├── src/                       # SABLE implementation and baseline models
├── experiments/
│   └── configs/               # Released experiment configurations
├── data/                      # Data placement instructions
└── LICENSE                    # MIT License
```

## Installation

The code is intended for **Python 3.9+** and **PyTorch 2.x**.

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

GPU acceleration is recommended for full training runs.

## Data

Data preparation and placement instructions are provided in [`data/README.md`](data/README.md).

The associated dataset record is available through IEEE DataPort:

**DOI:** `10.21227/d6ht-c375`

The dataset is subject to the terms of its original source and dataset record and is not covered by the MIT License of this repository.

## Quick Start

A lightweight configuration is provided for checking the environment and execution pipeline:

```bash
python main.py \
  --config experiments/configs/ablation/sable_quick_test.yaml \
  --device cpu
```

Alternatively, use the shell runner:

```bash
CONFIG=experiments/configs/ablation/sable_quick_test.yaml \
./run.sh --device cpu
```

The quick-test configuration is intended only for environment and pipeline validation and should not be interpreted as reproducing the numerical results reported in the manuscript.

## Experiment Configuration

The released YAML files under [`experiments/configs/`](experiments/configs/) document the configurations associated with the experiments.

The main SABLE configuration can be executed with:

```bash
python main.py \
  --config experiments/configs/ablation/sable_full.yaml
```

Please refer to the corresponding YAML files and the manuscript for detailed training settings, episodic protocols, evaluation procedures, and experimental definitions.

## Outputs

Depending on the selected configuration, runtime outputs may include logs, checkpoints, metrics, task-sampling records, and result files.

Generated outputs should be kept separate from archived experimental results.

## Citation

If you use this repository in your research, please cite the associated SABLE manuscript and, where applicable, the corresponding software release.

Citation information will be updated following publication of the manuscript.

## License

The source code in this repository is released under the [MIT License](LICENSE).

Data and other third-party resources remain subject to their respective licenses and terms of use.
