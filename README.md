# SABLE: A Sparse Bayesian Meta-Learning Framework for Few-Shot Traffic Incident Recognition

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This release package provides the implementation of **SABLE**, a sparse Bayesian meta-learning framework for few-shot traffic incident recognition.

The package accompanies our manuscript submitted to *IEEE Open Journal of Intelligent Transportation Systems* and provides the SABLE source code, a main configuration, and a lightweight environment-check configuration.

## Overview

The release is intended to support inspection of the proposed method and execution under the documented configurations. Detailed methodological definitions, experimental protocols, and result analysis are provided in the associated manuscript.

## Project Structure

```text
.
├── main.py                    # Command-line entry point
├── run.sh                     # Shell
├── requirements.txt           # Python dependencies
├── src/                       # SABLE implementation
├── experiments/
│   └── configs/ablation/      # Main and quick-test configurations
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

The dataset is subject to the terms of its original source and dataset record and is not covered by the MIT License applicable to the source code in this release package.

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

The release contains the main SABLE configuration and a reduced quick-test configuration. The quick test is intended only to check the environment and execution pipeline.

The main SABLE configuration can be executed with:

```bash
python main.py \
  --config experiments/configs/ablation/sable_full.yaml
```

## Outputs

Depending on the selected configuration, runtime outputs may include logs, checkpoints, metrics, task-sampling records, and result files.

Generated outputs should be kept separate from archived experimental results.

## Citation

If you use this release package or its source code in your research, please cite the associated SABLE manuscript and, where applicable, the corresponding software release.

Citation information will be updated following publication of the manuscript.

## License

The source code in this release package is released under the [MIT License](LICENSE).

Data and other third-party resources remain subject to their respective licenses and terms of use.
