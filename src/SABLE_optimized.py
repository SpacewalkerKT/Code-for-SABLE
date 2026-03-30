# SABLE_optimized.py - Performanceance Optimized Version
# # File header fix: removed mistakenly inserted runwhen code
#
# IMPORTANT:
# 1. This file is based on SABLE.py, specifically for performance optimization
# 2. Original SABLE.py remains unchanged as the official version for SCI paper
# 3. All optimizations preserve program logic and model performance
# 4. Only optimize computational efficiency, memory usage, and GPU utilization
# 5. Anyconfiguration changes affecting results will be confirmed in advance
#
# Optimization Log:
# - [2025-01-03] Created optimized version, started performance analysis
#
# Original File Information:
# - Project: Meta-Learning (MAML) Traffic Anomaly Detection Research
# - Purpose: SCI paper experimental code
# - Features: K-shot anomaly detectionone_vs_restAndoc_mamlMode

import pandas as pd
import numpy as np
import torch
import when
import logging
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from functools import wraps
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
)
# optional dependency:OptunaLazy importPlaceholder to avoidError when not installed
optuna = None

class Trial:  # Minimal placeholder class
    def set_user_attr(self, *args, **kwargs):
        pass
    def report(self, *args, **kwargs):
        pass
    def should_prune(self) -> bool:
        return False

class TrialPruned(Exception):
    pass
import warnings
warnings.filterwarnings("ignore", ".*`torch.nn.utils.stateless.functional_call` is deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", ".*numpy.core.numeric is deprecated.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", ".*findfont.*", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import random
import os
import sys
import pickle
import hashlib
import json
import math
import ast
from collections import OrderedDict, defaultdict, Counter
from abc import ABC, abstractmethod
import itertools
import torch.nn.utils as utils
from torch.nn.utils import stateless

# Compatible with PyTorch 2.x functional_call location changePrefer using torch.func.functional_call
try:  # PyTorch 2.0+
    from torch.func import functional_call as _functional_call_compat
except Exception:
    try:
        from torch.nn.utils.stateless import functional_call as _functional_call_compat
    except Exception:
        _functional_call_compat = None
import matplotlib
import matplotlib.pyplot as plt
## Enforce ASCII/English plotting to avoid CJK font lookups and noise
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
# Reduce Matplotlib internal logs (e.g., findfont) to errors only
logging.getLogger('matplotlib').setLevel(logging.ERROR)
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, optional, Tuple, Type, Any

# ===== Performanceance Optimization Tools =====
import psutil
import gc
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Thread
import queue

# ===== Focal Loss Loss Function implementingation =====
# Reference: Lin et al. "Focal Loss for Dense Object Detection." ICCV 2017
# To solveClass imbalance problemThroughDynamic weightFocus onDifficult samples
class FocalLoss(nn.Module):
    """
    Focal Loss Loss function

    Formula: FL(p_t) = -α_t(1-p_t)^γ log(p_t)

    parameters:
        alpha (float): Alpha balance parameter for positive/negative class balancing (Default: 0.25)
        gamma (float): Gamma focusing parameter controlling attention to hard samples (Default: 2.0)
        reduction (str): Loss aggregation method ('mean', 'sum', or 'none')
        class_weights (Tensor): optional class weight tensor [weight_neg, weight_pos]
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = 'mean', class_weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        """
        forward pass

        parameters:
            inputs: Model output logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            focal loss value
        """
        # Calculate cross-entropy probability
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.class_weights)

        # Get prediction probability
        probs = F.softmax(inputs, dim=1)
        # Extract probability of true class p_t
        p_t = probs[range(len(targets)), targets]

        # Calculate focal loss: -α(1-p_t)^γ * log(p_t)
        # Note: ce_loss already is -log(p_t), so directly use
        focal_weight = (1 - p_t) ** self.gamma

        # Apply class weights if provided
        if self.class_weights is not None:
            #  Fix double weight bug: ce_loss already includes class_weightsCannot multiply againweight_t
            # ce_loss = F.cross_entropy(..., weight=self.class_weights) Already containsweight
            # Only need to apply focal_weight here
            focal_loss = focal_weight * ce_loss
        else:
            # Apply alpha balance parameter
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_weight * ce_loss

        # Aggregate loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ===== Focal Loss End of implementation =====

@dataclass
class Performanceancemetrics:
    """Performanceance metrics Data Class"""
    whenstamp: float
    cpu_percent: float
    memory_mb: float
    gpu_memory_mb: optional[float] = None
    gpu_utilization: optional[float] = None
    operation: str = ""

class PerformanceanceMonitor:
    """Performanceance Monitor for tracking optimization effects"""

    def __init__(self):
        self.metrics_history = []
        self.operation_whens = {}
        self.start_when = when.when()

    @contextmanager
    def whenr(self, operation_name: str):
        """Timing Context Manager"""
        start_when = when.when()
        try:
            yield
        finally:
            elapsed = when.when() - start_when
            self.operation_whens[operation_name] = elapsed
            print(f"[PerfMon] {operation_name}: {elapsed:.4f}sec")

    def record_metrics(self, operation: str = ""):
        """Record currentSystem performance metrics"""
        metrics = Performanceancemetrics(
            whenstamp=when.when(),
            cpu_percent=psutil.cpu_percent(),
            memory_mb=psutil.virtual_memory().used / 1024 / 1024,
            operation=operation
        )

        # GPU monitoring (if available)
        if torch.cuda.is_available():
            metrics.gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            # NOTE: GPU utilization requires nvidia-ml-py, here temporarily no dependencies on

        self.metrics_history.append(metrics)
        return metrics

    def print_summary(self):
        """Print performance summary"""
        total_when = when.when() - self.start_when
        print("\n" + "="*60)
        print("Performanceance monitoring summary")
        print("="*60)
        print(f"Total runwhen: {total_when:.2f}sec")

        if self.operation_whens:
            print("\nwhen per operation:")
            for op, when_cost in sorted(self.operation_whens.items(),
                                       key=lambda x: x[1], reverse=True):
                print(f"  {op}: {when_cost:.4f}sec")

        if self.metrics_history:
            print(f"\n Memory usage peak value: {max(m.memory_mb for m in self.metrics_history):.2f}MB")
            if torch.cuda.is_available():
                gpu_peak = max(m.gpu_memory_mb for m in self.metrics_history
                             if m.gpu_memory_mb)
                print(f"GPU memory peak value: {gpu_peak:.2f}MB")
        print("="*60)

# Global performance monitor instance
perf_monitor = PerformanceanceMonitor()

# ===== Performanceance Optimization ToolsFinished =====

# ===== StructuredprogresswithResultsExportMinimal enhancement =====
# NOTE
# - Support atRunStartingDuring trainingbased onIntervalWrite when finished JSON lineFor assistant/Script parsing
# - Configuration pathPriorityanalysis_export.progress_jsonl > analysis_export.metrics_jsonl > runs/<ts>/metrics.jsonl
PROGRESS_JSONL_PATH: optional[str] = None
PROGRESS_EVAL_INTERVAL: int = 50  # Default every 50 roundWriteOnceprogress

def _ensure_parent_dir(path: str) -> None:
    if not path:
        return
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _append_jsonl(path: optional[str], record: Dict) -> None:
    if not path:
        return
    try:
        _ensure_parent_dir(path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:  # pragma: no cover
        print(f"[Progress] Write {path} Failed: {e}")

def report_start(seed: optional[int], episodes_total: optional[int], config_path: optional[str], run_dir: optional[str]):
    rec = {
        "status": "start",
        "whenstamp": when.strfwhen('%Y-%m-%d %H:%M:%S'),
        "seed": int(seed) if isinstance(seed, int) else seed,
        "episodes_total": int(episodes_total) if isinstance(episodes_total, (int, float)) else episodes_total,
        "config": config_path,
        "run_dir": run_dir,
    }
    _append_jsonl(PROGRESS_JSONL_PATH, rec)

def report_progress(episode: int, episodes_total: int, elapsed_sec: float, metrics: optional[Dict] = None,
                    config_path: optional[str] = None, run_dir: optional[str] = None):
    rec = {
        "status": "progress",
        "whenstamp": when.strfwhen('%Y-%m-%d %H:%M:%S'),
        "episode": int(episode),
        "episodes_total": int(episodes_total),
        "when_sec": float(elapsed_sec),
        "config": config_path,
        "run_dir": run_dir,
    }
    if isinstance(metrics, dict):
        # Only mergeCommon metrics keysAvoid writing too large objects
        for k in ("f1", "recall", "precision", "accuracy", "auc", "threshold"):
            if k in metrics:
                rec[k] = metrics[k]
    _append_jsonl(PROGRESS_JSONL_PATH, rec)

# ===== StructuredprogressExportFinished =====

# optionally disable plottingFor serverAvoid long when in validation phase/Block
# Controlled by environment variableSABLE_DISABLE_PLOTS=1 Will skip allPlotting and saving
DISABLE_PLOTTING = str(os.environ.get("SABLE_DISABLE_PLOTS", "0")).lower() in ("1", "true", "yes")

# On display-less server prefer using Agg backendAvoid problems caused by interactive backend
try:
    if not matplotlib.get_backend().lower().startswith("agg"):
        matplotlib.use("Agg", force=True)
except Exception:
    pass

# Set Chinese fontAvoid missing Chinese display
plt.rcParams['font.sans-serif'] = ['SimHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False
# Fallback: If train entry not implementedProvide explicit placeholder functionPrevent undefined name
if 'run_training_trial' not in globals():
    def run_training_trial(params):  # pragma: no cover
        raise NotimplementingedError(
            "run_training_trial Not yet in SABLE_optimized.py implementinged inPlease integrateOriginal training process"
            "to avoidOptuna is not installed Optuna/not integratedtraining entry is not integrated, calling parameter tuning logic"
        )


# ===== Log toolFor print OutputRedirect to logging =====
class StreamToLogger:
    """Pseudo file stream object, for standard output or error redirect to logging recorder"""

    def __init__(self, logger, level):
        """Initialize redirectorRecord target recorder with log level"""
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        """Write cache line by line and send to corresponding log level"""
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        """Compatible withfileinterfaceofFlush methodNo operation needed here"""
        pass


# =====configurationperformance optimizationVersion =====
K_SHOT = 5
Q_SHOT = 10
EPISODES = 5000
SEED = 42
patience = 500  # define "patience", i.e., when validation set performance does not improveMaximum rounds to wait
lr_inner = 0.005
lr_meta = 0.01
kl_warmup_steps = 0  # KL divergence warmup steps0 means according to total rounds 20% automatically calculate
kl_scale = 0.01  # KL divergence loss scaling factorAligned with reported default λ_KL
inner_steps = 5
OTHER_ABNORMAL_RATIO = 0.5  # Other abnormal samples ratio in negative class
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Script located directory
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")  # Checkpoint save directory
SAVE_EVERY = 50  # Every50episodeSaveOnce

# ===== performance optimizationconfiguration =====
# TrainingPerformanceanceRelated
ENABLE_PERFORMANCE_MONITORING = True  # Enable performance monitoring
USE_MIXED_PRECISION = False  # mixed precision trainingRequires CUDAsupport
GRAD_CLIP_ENABLED = True  # Enable gradient clipping
GRAD_CLIP_VALUE = 1.0  # Gradient clipping value

# memory optimizationconfiguration
PIN_MEMORY = True  # whether DataLoader pins memory
NUM_WORKERS = 0  # Data loading worker processes0 means main process
CACHE_SIZE_LIMIT = 1000  # CacheSizeLimit
ENABLE_GARBAGE_COLLECTION = True  # enable periodic garbage collection
GC_FREQUENCY = 10  # Every10episodeExecuteOnceGarbage collection

# batchProcessOptimize
DYNAMIC_BATCH_SIZE = False  # Whether to dynamically adjust batch size
MINI_BATCH_ENABLED = False  # Whether to use small batch accumulate gradient
ACCUMULATION_STEPS = 1  # Gradient accumulation steps

# AlgorithmsOptimize
USE_EARLY_STOPPING = True  # enabledearly stopping
VALIDATION_FREQUENCY = 10  # ValidationFrequency
LOG_FREQUENCY = 1  # logOutputFrequency

print(f"[performance optimization] Performanceancemonitor: {'enabled' if ENABLE_PERFORMANCE_MONITORING else 'disabled'}")
print(f"[performance optimization] Garbage collection: {'enabled' if ENABLE_GARBAGE_COLLECTION else 'disabled'} (Frequency: Every{GC_FREQUENCY}round)")

# ===== recall priority and threshold schedulingconfigurationnewly added =====
# NOTE: Use this parameter during validation phase based on "recall floor + F_beta maximization" select thresholds, and use for early stopping / optimal model selection with test set frozen threshold value evaluation.
RMIN_SCHEDULE = "linear"  # optional"linear"linear threshold increase/ "ladder"adaptive ladder (backup)
R_START = 0.80            # recall floorstarting valuealigned with report“code default fixed value 0.80”for alignment
R_TARGET = 0.80           # targetrecall floorconsistent with startthereby implementingfixed R_min
R_WARMUP_FRAC = 0.0       # fixed R_min set to 0linear degrades to constant R_min=0.80

FBETA_BETA = 2.0          # F_beta of β parameterβ=2 More weight on recallRecommended
THRESH_EMA = 0.2          # threshold exponentialmoving average coefficientsmooth fluctuations between evaluations
POS_RATE_CAP = 0.90       # threshold for excessive positive class predictionsTighten to suppress positive bias

# adaptive ladder threshold increasebackup switch(when enabled)
LADDER_WINDOW = 5         # sliding window over recentsliding window
LADDER_MARGIN = 0.03      # when recall withinis generally higher than (current R_min + this margin) only adjust
LADDER_STEP = 0.02        # adjustment step size

# TrainingOutputwithValidationEvaluationFrequencyDecouple original report_every
LOG_EVERY = 1             # Traininging side log print frequency: print every episode
EVAL_EVERY = 10           # validation and early stopping check frequencyEvery10roundExecuteOnce

# ===== feature modeconfigurationTo facilitate switching 75-dim and 25-dimExperiments =====
FEATURE_MODE = os.environ.get("MAML_FEATURE_MODE", "volume_only")  # optional"full""volume_only"
_FEATURE_TYPE_MAP = {
    "full": ["Occupancy_Rate", "Speed", "Volume"],
    "volume_only": ["Volume"],
}

if FEATURE_MODE not in _FEATURE_TYPE_MAP:
    raise ValueError("Invalid FEATURE_MODE, please choose between 'full' and 'volume_only' BetweenSelect")

FEATURE_TYPES = _FEATURE_TYPE_MAP[FEATURE_MODE]
FEATURE_CHANNELS = len(FEATURE_TYPES)
FEATURE_DIM = FEATURE_CHANNELS * 25

print(f"[Feature Config] Current mode = {FEATURE_MODE}Sensor types = {FEATURE_CHANNELS}Inputdimension = {FEATURE_DIM}")

# ===== class weight fixed switchTo emphasize recall =====
USE_CLASS_WEIGHT = True      # whether to enable class weightingcore SABLE functionality
USE_FIXED_POS_WEIGHT = True  # fixed positive weight rather thanbased onper-query count
FIXED_POS_WEIGHT = 1.6       # fixed positive class weight valuealigned with optimal w_pos=1.6 for alignment

# ===== Threshold searchswitchablation experiment control =====
USE_THRESHOLD_SEARCH = True  # whether to use adaptive threshold searchFalse(when enabled)fixed threshold 0.5
FIXED_THRESHOLD = 0.5        # disabledThreshold search(when enabled)offixed threshold

# ===== hard negative sampling strategycore functionality =====
HARD_NEGATIVE_FLAG = True  # whether to enable hard negative sampling strategycore strategy, enabled by default

# ===== Sparse variational dropoutcomponentArchitectural Innovations =====
# Added config reading support, fixed P4 ablation experiment invalidation problem
USE_SPARSE_VD = True  # whether to use SparseVD linear layerTrue: B_SMALL_Transformer, False: StandardTransformer

# ===== Meta-learningFrameworkcomponentArchitectural Innovations =====
# Addedconfiguration reading support, fixed P6 ablation experiment invalidation problem
USE_META_LEARNING = True  # whether to use MAML meta-learning frameworkTrue: train_maml, False: standard supervised learning


def objective(trial: Trial) -> float:
    """Optuna target functionatsoft recall constraintunderfind optimalhyperparameters."""

    # ===== 1. define necessary explorationofhyperparameter =====
    params: Dict[str, float] = {
        "inner_lr": trial.suggest_float("inner_lr", 1e-4, 1e-1, log=True),
        "outer_lr": trial.suggest_float("outer_lr", 1e-5, 1e-2, log=True),
        "grad_clip_norm": trial.suggest_float("grad_clip_norm", 0.1, 10.0),
        "positive_weight": trial.suggest_float("positive_weight", 1.0, 20.0),
    }
    trial.set_user_attr("suggested_params", params)

    # ===== 2. definesoft constraintconstraint withpruning callback =====
    RECALL_MIN_THRESHOLD = 0.55
    PENALTY_LAMBDA = 2.0

    def _pruning_callback(step_index: int, metrics: optional[Dict[str, float]] = None) -> None:
        """for training process in callbackpruning hook forbased on soft constraint scoremake early stopping deccision"""
        if metrics is None:
            return
        recall_val = float(metrics.get("recall", 0.0))
        f1_val = float(metrics.get("f1", 0.0))
        recall_gap = recall_val - RECALL_MIN_THRESHOLD
        penalty_val = PENALTY_LAMBDA * max(0.0, -recall_gap)
        intermediate_score = f1_val - penalty_val
        trial.report(intermediate_score, step=int(step_index))
        if trial.should_prune():
            raise TrialPruned(
                f"trial No. {step_index} monitor point pruned byintermediate score {intermediate_score:.4f}"
            )

    params_for_training = dict(params)
    params_for_training["optuna_trial"] = trial
    params_for_training["pruning_callback"] = _pruning_callback

    # ===== 3. RunOncecompletedtraining processInternalfixedRandom seed =====
    try:
        training_result = run_training_trial(params_for_training)
        trial.set_user_attr("pruning_callback_attached", True)
    except TypeError as exc:
        # compatible with old version run_training_trial does not accept additional parameterscase
        if "unexpected keyword argument" not in str(exc):
            raise
        training_result = run_training_trial(params)
        trial.set_user_attr("pruning_callback_attached", False)

    if isinstance(training_result, dict):
        y_true = training_result.get("y_true")
        y_scores = training_result.get("y_scores")
    else:
        y_true, y_scores = training_result

    if y_true is None or y_scores is None:
        raise ValueError("run_training_trial must return validation set labels and predictionsionprobability")

    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_scores)

    thresholds = np.linspace(0.01, 0.99, 100)
    best_score = -np.inf
    best_threshold = float(thresholds[0])
    best_f1 = 0.0
    best_recall = 0.0
    best_precision = 0.0
    best_recall_gap = -np.inf

    # ===== 4. exhaustively trythreshold valuescalculate penalized score =====
    for threshold in thresholds:
        predictions = (y_score_arr >= threshold).astype(int)
        recall_val = recall_score(y_true_arr, predictions, zero_division=0)
        precision_val = precision_score(y_true_arr, predictions, zero_division=0)
        f1_val = f1_score(y_true_arr, predictions, zero_division=0)

        recall_gap = recall_val - RECALL_MIN_THRESHOLD
        penalty = PENALTY_LAMBDA * max(0.0, -recall_gap)
        current_score = f1_val - penalty

        if current_score > best_score:
            best_score = current_score
            best_threshold = float(threshold)
            best_f1 = float(f1_val)
            best_recall = float(recall_val)
            best_precision = float(precision_val)
            best_recall_gap = float(recall_gap)

    # ===== 5. record auxiliary infofacilitate subsequent analysis =====
    penalty_final = PENALTY_LAMBDA * max(0.0, -best_recall_gap)
    trial.set_user_attr("best_threshold", best_threshold)
    trial.set_user_attr("best_f1", best_f1)
    trial.set_user_attr("best_recall", best_recall)
    trial.set_user_attr("best_precision", best_precision)
    trial.set_user_attr("recall_gap", best_recall_gap)
    trial.set_user_attr("penalty", penalty_final)
    trial.report(best_score, step=len(thresholds))

    return float(best_score)


def run_optuna_hyperparameter_search(n_trials: int = 100):  # pragma: no cover
    """placeholder:hyperparameter searchcurrentminimal run pathnot mandatory
    If enabled, install optuna and integrate objective function in training entryy
    """
    raise NotImplementedError("Current optimized version does not enable hyperparameter tuning; please implement when necessary; remove this placeholder.")


'''
Every50episodeAutomaticSavecheckpointto checkpoints/ Directory
manually specify resume pointset resume_from='checkpoint_1000.pth'
Start from beginning: delete checkpoints/ directory or select 'no resume' at promptResume
checkpoint cleanupautomatically keeprecent3checkpointdelete old to save space
EventuallymodelsSaveTraininging completeafterSaveEventuallymodelsfor final_model.pth
'''

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SANITY = False  # True willrun fewer episodes for quick check
# ===== cache and checkpointmaster switch =====
# USE_CACHE for True whenwill prioritize loadingexceptiontask dictionarynormal pools and resampled training poolsfrom cache
# If cache signature matches currentconfiguration, directly reuse cached  and enter training phase, no need to reo resampling
# when cache missingor datachangeswill only rescanoriginal CSV and execute full buildwith resampling process
USE_CACHE = True  # False indicates disabled cachealways rebuildtask dictionarywithnormal pools
USE_CHECKPOINT = False  # True whenAutomaticTryfromcheckpointResumeTrainingprogress

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
session_start_when = when.when()

# ensure checkpoint directoryexists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ===== CacheRelatedset =====
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
NORMAL_POOL_CACHE = os.path.join(CACHE_DIR, "normal_pool.pkl")
TASK_DICT_CACHE = os.path.join(CACHE_DIR, "task_dict.pkl")
RESAMPLED_TRAIN_CACHE = os.path.join(CACHE_DIR, "resampled_train_pool.pkl")

# ===== Support Set Standardization Statistics Cache =====
PROC_INFO_RUNTIME = {
    "zero_counts": Counter(),
    "perturb_counts": Counter(),
    "records": 0,
}


def safe_read_csv(path, usecols=None, dtype_map=None, chunk_size=50000):
    """
    optimized versionread CSV filewhen out of memoryautomatic chunked reading
    Optimization points:
    1. increase chunk_sizeto50000to reduce I/O count
    2. use more efficientconcatenation strategy
    3. Addprogressdisplay
    """
    try:
        with performance monitor whenr(f"Read{os.path.basename(path)}"):
            return pd.read_csv(path, usecols=usecols, dtype=dtype_map)
    except (pd.errors.ParserError, MemoryError):
        print(f"[performance optimization] {path} file too largeuse optimized chunked reading (chunk_size={chunk_size})...")
        chunks = []
        total_rows = 0

        with performance monitor whenr(f"chunked reading{os.path.basename(path)}"):
            try:
                for i, chunk in enumerate(pd.read_csv(path, usecols=usecols, dtype=dtype_map, chunksize=chunk_size)):
                    chunks.append(chunk)
                    total_rows += len(chunk)
                    if i % 5 == 0:  # Every5chunkdisplayOnceprogress
                        print(f"[Progress] Read {total_rows} rows...")

                df = pd.concat(chunks, ignore_index=True, copy=False)
                print(f"[completed] chunked readingcompletedtotal rows: {len(df)}")
                return df

            except MemoryError:
                # If still out of memory, use smaller chunks
                smaller_chunk = chunk_size // 5
                print(f"[memory optimization] Memorystill insufficientuse smaller chunks {smaller_chunk}")
                chunks = []
                for chunk in pd.read_csv(path, usecols=usecols, dtype=dtype_map, chunksize=smaller_chunk):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True, copy=False)
                return df


# ===== DataConstruct =====
class BaselineStatsTracker:
    """for monitoringbaseline differencevariance utility class"""

    def __init__(self, name: str,
                 delta_threshold: float = 1e-3,
                 baseline_threshold: float = 1e-4):
        # Monitor object name (e.g., normal pool or anomaly samples)
        self.name = name
        # difference sequencelow variance threshold
        self.delta_threshold = float(delta_threshold)
        # baseline segmentlow variance threshold
        self.baseline_threshold = float(baseline_threshold)
        # record each sampleof difference variance
        self.delta_vars: List[float] = []
        # record baseline segmentvariance offacilitate decide if baselineis nearly constant
        self.baseline_vars: List[float] = []

    def update(self, baseline_segment: np.ndarray, delta_series: np.ndarray):
        """Updatemonitorstatistics"""
        if baseline_segment is None or delta_series is None:
            return
        # calculate baseline segmentwith difference sequencevariance of
        baseline_var = float(np.var(baseline_segment))
        delta_var = float(np.var(delta_series))
        self.baseline_vars.append(baseline_var)
        self.delta_vars.append(delta_var)

    def summary(self) -> Dict[str, float]:
        """GeneratestatisticsAbstractForlogrecordwiththreshold valuedecision"""
        if not self.delta_vars:
            return {
                "name": self.name,
                "count": 0,
                "recommend_change": False,
                "reason": "No available samples"
            }

        delta_arr = np.array(self.delta_vars, dtype=float)
        base_arr = np.array(self.baseline_vars, dtype=float)

        low_delta_ratio = float(np.mean(delta_arr < self.delta_threshold))
        low_base_ratio = float(np.mean(base_arr < self.baseline_threshold))

        recommend = False
        reasons: List[str] = []
        if low_delta_ratio > 0.4:
            recommend = True
            reasons.append(f"difference value variancelower than {self.delta_threshold:.1e} of sample ratio {low_delta_ratio * 100:.1f}%")
        if delta_arr.mean() < self.delta_threshold:
            recommend = True
            reasons.append(f"difference value variancemean value {delta_arr.mean():.1e} too small")
        if low_base_ratio > 0.4:
            recommend = True
            reasons.append(f"baseline segment variancelower than {self.baseline_threshold:.1e} of sample ratio {low_base_ratio * 100:.1f}%")

        summary = {
            "name": self.name,
            "count": int(len(delta_arr)),
            "delta_var_mean": float(delta_arr.mean()),
            "delta_var_median": float(np.median(delta_arr)),
            "delta_var_min": float(delta_arr.min()),
            "delta_var_max": float(delta_arr.max()),
            "low_delta_ratio": low_delta_ratio,
            "baseline_var_mean": float(base_arr.mean()),
            "baseline_var_median": float(np.median(base_arr)),
            "baseline_var_min": float(base_arr.min()),
            "baseline_var_max": float(base_arr.max()),
            "low_baseline_ratio": low_base_ratio,
            "delta_threshold": self.delta_threshold,
            "baseline_threshold": self.baseline_threshold,
            "recommend_change": recommend,
            "reason": "".join(reasons) if reasons else ""
        }
        return summary


def _print_baseline_summary(summary: Dict[str, float]):
    """Unified output of monitoring results."""
    if not summary or summary.get("count", 0) == 0:
        print(f"baseline differencestatistics[{summary.get('name', 'unknown')}]: no available samples")
        return
    name = summary["name"]
    print(
        f"Baseline difference statistics[{name}]: Sample count={summary['count']}, "
        f"difference value variancemean value={summary['delta_var_mean']:.3e}, median={summary['delta_var_median']:.3e}, "
        f"low varianceratio={summary['low_delta_ratio'] * 100:.2f}%"
    )
    print(
        f"  baseline segment variancemean value={summary['baseline_var_mean']:.3e}, low varianceratio={summary['low_baseline_ratio'] * 100:.2f}%"
    )
    if summary.get("recommend_change"):
        reason = summary.get("reason", "") or "difference value with baselinevariance too small"
        print(f"   recommend re-evaluatingbaseline calculation logic{reason}")


def build_normal_pool(normal_df, after_prob=0.5, tracker: optional[BaselineStatsTracker] = None):
    """Build feature pool of normal sample differences and preserve metadata."""
    pool: List[Dict[str, object]] = []
    grouped = normal_df.groupby(["IncidentId", "SensorNumber"])
    data_cols = [f"Data{i}" for i in range(1, 26)]

    for (iid, sid), group in grouped:
        delta_segs = []
        kind_summary = []  # record before/after selection for each type

        for dtype in FEATURE_TYPES:
            # Modifyadapt to new data formatonly after datano Kind column
            # check if there is Kind column
            if "Kind" in group.columns:
                # Old data format: has before and after
                before = group[(group["Data_Type"] == dtype) & (group["Kind"] == "before")]
                after = group[(group["Data_Type"] == dtype) & (group["Kind"] == "after")]
                if before.shape[0] != 1 or after.shape[0] != 1:
                    delta_segs = []
                    break

                try:
                    b = before[data_cols].astype(float).values[0]
                    a = after[data_cols].astype(float).values[0]

                    if random.random() < after_prob:
                        selected_data = a
                        selected_row = after.iloc[0]
                        selected_kind = "after"
                    else:
                        selected_data = b
                        selected_row = before.iloc[0]
                        selected_kind = "before"

                    if not np.isfinite(selected_data).all() or selected_data.shape[0] != 25:
                        delta_segs = []
                        break

                    baseline_data = selected_data[:5]
                    baseline = np.mean(baseline_data)
                    delta_series = selected_data - baseline

                    if tracker is not None:
                        tracker.update(baseline_data, delta_series)

                    if not np.isfinite(delta_series).all():
                        delta_segs = []
                        break

                    delta_segs.append(delta_series)
                    kind_summary.append((dtype, selected_kind))  # keep tuple formatwithNo.817lineconsistent

                except Exception as e:
                    # improverecordexceptionINFOfacilitateDEBUG
                    logging.warning(f"error processing old data format (IncidentId={iid}, SensorNumber={sid}, dtype={dtype}): {e}")
                    delta_segs = []
                    break
            else:
                # new data formatonly after data
                data_rows = group[group["Data_Type"] == dtype]
                if data_rows.shape[0] == 0:
                    delta_segs = []
                    break

                try:
                    # directly use firstitems after data
                    selected_data = data_rows[data_cols].astype(float).values[0]
                    selected_row = data_rows.iloc[0]

                    if not np.isfinite(selected_data).all() or selected_data.shape[0] != 25:
                        delta_segs = []
                        break

                    baseline_data = selected_data[:5]
                    baseline = np.mean(baseline_data)
                    delta_series = selected_data - baseline

                    if tracker is not None:
                        tracker.update(baseline_data, delta_series)

                    if not np.isfinite(delta_series).all():
                        delta_segs = []
                        break

                    delta_segs.append(delta_series)
                    kind_summary.append((dtype, "after"))  # new data is all afterkeep tuple format

                except Exception as e:
                    # improverecordexceptionINFOfacilitateDEBUG
                    logging.warning(f"error processing new data format (IncidentId={iid}, SensorNumber={sid}, dtype={dtype}): {e}")
                    delta_segs = []
                    break

        if len(delta_segs) == FEATURE_CHANNELS:
            if FEATURE_CHANNELS == 1:
                vec = delta_segs[0]
            else:
                vec = np.concatenate(delta_segs)
            if vec.shape == (FEATURE_DIM,) and np.isfinite(vec).all():
                meta_row = group.iloc[0]
                entry: Dict[str, object] = {
                    "features": vec.astype(float),
                    "IncidentId": int(iid),
                    "SensorNumber": int(sid),
                }
                # Extract common metadata columns (if exists)
                for col in ["Week", "tag", "DESCRIPTION", "Percentage", "Median"]:
                    if col in meta_row:
                        value = meta_row[col]
                        if isinstance(value, (np.generic,)):
                            value = value.item()
                        entry[col] = value
                entry["kind_mix"] = {dtype: kind for dtype, kind in kind_summary}
                pool.append(entry)
    return pool


def extract_abnormal_features(df, incident_id, sensor_id,
                              tracker: optional[BaselineStatsTracker] = None):
    """
    optimized versionExtractexceptionDataofdifference valuewhen sequenceFeature

    Optimization points:
    1. pre-generatedata column namesavoid repeated creation
    2. use vectorized operationsreplace loops
    3. optimize memory allocationand type conversion
    4. add early exit conditions

    Step1CalculateEventbeforeBaselineData1-Data5 ofmean value
    Step2calculate difference valueswhen sequenceData1-Data25 minus baseline
    Step 3: based onfeature type concatenationfor final vectordimension = FEATURE_DIM

    Args:
        df: exceptionData DataFrame
        incident_id: event ID
        sensor_id: sensor ID
        tracker: optional baseline variance monitorfor recording difference value variance distribution
    if the same (IncidentId, SensorNumber, Data_Type) has multiple rows: prioritize based on time/index, take most recent item; find column then take first itemms
    """
    # optimization: pre-allocateresults array
    delta_segs = np.empty(len(FEATURE_TYPES) * 25, dtype=np.float32)

    # Optimizepre-generatedata column names
    data_cols = [f"Data{i}" for i in range(1, 26)]

    for idx, t in enumerate(FEATURE_TYPES):
        # Optimizeuse more efficient booleanIndex
        mask = (df["IncidentId"].values == incident_id) & \
               (df["SensorNumber"].values == sensor_id) & \
               (df["Data_Type"].values == t)

        sub = df[mask]
        if sub.shape[0] == 0:
            return None

        # select single item record: optimize column check logic (ensure obtained is“single row of Seriess”
        if "when" in sub.columns:
            sub = sub.iloc[int(sub["when"].values.argmax())]
        elif "row_index" in sub.columns:
            sub = sub.iloc[int(sub["row_index"].values.argmax())]
        else:
            sub = sub.iloc[0]

        try:
            # take single from Series extract 25-dimensional vectoravoid Series treating as DataFrame leading to 2D->1D then [0] error
            if hasattr(sub, "__getitem__") and not isinstance(sub, pd.DataFrame):
                row_series = sub
            else:
                row_series = sub.iloc[0]
            vals = row_series[data_cols].to_numpy(dtype=np.float32, copy=False)
        except (IndexError, valueError, KeyError, AttributeError):
            return None

        # Optimization: fast validity check
        if not np.isfinite(vals).all():
            return None

        # Step1CalculateEventbeforeBaselineData1-Data5ofmean value
        # Optimizeuse built-in functionand pre-slicing
        baseline = float(np.mean(vals[:5], dtype=np.float32))

        # Step2calculate difference  sequence (all 25 time points minus baseline)
        # Optimizedirectly write to pre-allocated array
        start_idx = idx * 25
        delta_series = vals - baseline
        delta_segs[start_idx:start_idx + 25] = delta_series

        if tracker is not None:
            # Optimizedirectly use baseline dataand difference sequence
            tracker.update(vals[:5], delta_series)

        # check difference sequenceis valid
        if not np.isfinite(delta_series).all():
            return None

    # Step3return final featurevectoralready in pre-allocated array
    return delta_segs if delta_segs.shape == (FEATURE_DIM,) and np.isfinite(delta_segs).all() else None


def build_abnormal_tasks(incident_df, tracker: optional[BaselineStatsTracker] = None):
    """based on Description+Week+tag into exception typesonly keep successfully extracted to feature pairs

    Args:
        incident_df: original anomaly sample table.
        tracker: optional baseline variance monitorfor recording difference value variance distribution
    """
    incident_df = incident_df.copy()
    # use vectorized string concatenationgenerate task identifiersavoid row-wise apply causing high memory overhead
    incident_df["task"] = (
            incident_df["DESCRIPTION"].astype(str)
            + "__Week"
            + incident_df["Week"].astype(str)
            + "__tag"
            + incident_df["tag"].astype(str)
    )

    # first aggregate all (task, IncidentId, SensorNumber)
    grouped = incident_df.groupby(["task", "IncidentId", "SensorNumber"], as_index=False).size()
    grouped = grouped[["task", "IncidentId", "SensorNumber"]]

    task_dict = defaultdict(list)
    # per pair attempt to extract featuresonly count if successful
    for _, row in grouped.iterrows():
        k = row["task"];
        iid = row["IncidentId"];
        sid = row["SensorNumber"]
        f = extract_abnormal_features(incident_df, iid, sid, tracker)
        if f is not None and f.shape == (FEATURE_DIM,) and np.isfinite(f).all():
            task_dict[k].append((iid, sid))

    # only keep samples withsufficient tasks
    keep = {k: v for k, v in task_dict.items() if len(v) >= (K_SHOT + Q_SHOT)}
    return keep


# ===== Feature Validation and Standardization =====
def is_valid(vec, expected_len):
    return isinstance(vec, np.ndarray) and vec.shape == (expected_len,)


def _record_proc_info(proc_info: Dict[str, List[int]]):
    """record standardization phase of zero variancewith low variancecolumns for offline statistics"""
    if proc_info is None:
        return

    zero_cols = (
            proc_info.get("zero_var_cols_reinserted")
            or proc_info.get("zero_var_cols")
            or []
    )
    perturbed_cols = (
            proc_info.get("low_var_cols_perturbed")
            or proc_info.get("perturbed_cols")
            or []
    )

    PROC_INFO_RUNTIME["records"] += 1
    if zero_cols:
        PROC_INFO_RUNTIME["zero_counts"].update(int(col) for col in zero_cols)
    if perturbed_cols:
        PROC_INFO_RUNTIME["perturb_counts"].update(int(col) for col in perturbed_cols)


def check_support_quality(support_x):
    """checksupport set forbasic qualityreturn discovered problematic variances"""
    issues = {}
    ifsupport_x.shape[0] < 2:
        issues["too_few_samples"] = int(support_x.shape[0])

    # calculate each column variance for subsequent analysis
    variances = np.var(support_x, axis=0)

    # find zero varianceand too small variance of columns
    zero_var_cols = np.where(variances == 0)[0]
    low_var_cols = np.where((variances > 0) & (variances < 1e-8))[0]
    if zero_var_cols.size > 0:
        issues["zero_var_cols"] = zero_var_cols.tolist()
    if low_var_cols.size > 0:
        issues["low_var_cols"] = low_var_cols.tolist()

    return issues, variances


def standardize_with_support(support_x, query_x):
    """
    optimized version: only useSupport setfit scaler and apply to query setat the same time detect potentialntial data problemsem

    Optimization points:
    1. pre-allocateresults arrayavoid repeated creation
    2. optimize memory layoutand array operations
    3. Reduce unnecessary array copies
    4. Use more efficient numerical operations
    """
    issues, variances = check_support_quality(support_x)
    if "too_few_samples" in issues:
        print(f"WARNING:support set sample counttoo few: {issues['too_few_samples']}")

    zero_var_cols = sorted(issues.get("zero_var_cols", []))
    low_var_cols = sorted(issues.get("low_var_cols", []))

    original_dim =support_x.shape[1]
    # Optimizeuse numpy ones to directly createboolean mask
    keep_mask = np.ones(original_dim, dtype=bool)

    if zero_var_cols:
        print(f"WARNING:support set containszero variance offeature columns: {zero_var_cols}standardization temporarilynRemove")
        keep_mask[zero_var_cols] = False

    # Optimizeavoid unnecessary copiesuse views
   support_reduced =support_x[:, keep_mask]
    query_reduced = query_x[:, keep_mask]

    ifsupport_reduced.shape[1] == 0:
        print("WARNING: all features have zero variancestandardization results willall be zero")
        # Optimization: pre-allocate zero array with correct data type
        sx = np.zeros_like(support_x, dtype=np.float32)
        qx = np.zeros_like(query_x, dtype=np.float32)
        leak_warning = False
        proc_info = {
            "zero_var_cols_reinserted": zero_var_cols,
            "low_var_cols_perturbed": [],
            "leak_warning": leak_warning,
        }
        _record_proc_info(proc_info)
        return sx, qx, proc_info

    # Optimizesimplify low variance columnprocessing logic
    perturbed_cols_original = []
    if low_var_cols:
        # Optimizedirectly calculate necessaryperturbed column indices
        current_indices = np.where(keep_mask)[0]
        valid_low_var_cols = [col for col in low_var_cols if col in current_indices]
        if valid_low_var_cols:
            print(f"Tip: for low variance columns {valid_low_var_cols} add perturbation1e-8")
            # Optimizeuse advanced index batchoperations
            col_positions = np.searchsorted(current_indices, valid_low_var_cols)
           support_reduced[:, col_positions] += 1e-8
            query_reduced[:, col_positions] += 1e-8
            perturbed_cols_original = valid_low_var_cols

    # Standardization process
    scaler = StandardScaler(with_mean=True, with_std=True)
    # Optimizedirectly pass, avoid intermediate variables
    sx_reduced = scaler.fit_transform(support_reduced)
    qx_reduced = scaler.transform(query_reduced)

    # Optimizesimplify data leakage detection
    # sample a small subset of datafor quick checkinstead of calculating all data
    combined_sample = np.vstack([support_reduced[:min(5, len(support_reduced))],
                                query_reduced[:min(5, len(query_reduced))]])
    scaler_all = StandardScaler(with_mean=True, with_std=True).fit(combined_sample)
    leak_warning = np.allclose(scaler.mean_, scaler_all.mean_, rtol=1e-3) and \
                   np.allclose(scaler.scale_, scaler_all.scale_, rtol=1e-3)

    if leak_warning:
        print("WARNING:support set with complete datafitted parameters are consistentdata leakage may exist")

    # Optimization: fast validation (only in development mode)
    if __debug__ and sx_reduced.shape[1] > 0:
        assert np.allclose(sx_reduced.mean(axis=0), 0, atol=1e-5)
        assert np.allclose(sx_reduced.std(axis=0), 1, atol=1e-5)

    # Optimizeefficiently reconstructoriginal dimension array
    if zero_var_cols:
        # Optimizeuse in-place operations to avoidadditional copies
        sx = np.zeros(support_x.shape, dtype=sx_reduced.dtype)
        qx = np.zeros(query_x.shape, dtype=qx_reduced.dtype)
        sx[:, keep_mask] = sx_reduced
        qx[:, keep_mask] = qx_reduced
    else:
        sx, qx = sx_reduced, qx_reduced

    # Optimizein-place denoising operations
    np.nan_to_num(sx, copy=False)
    np.nan_to_num(qx, copy=False)

    # Return process info for caller to log
    proc_info = {
        "zero_var_cols_reinserted": zero_var_cols,
        "low_var_cols_perturbed": perturbed_cols_original,
        "leak_warning": leak_warning,
    }
    _record_proc_info(proc_info)
    return sx, qx, proc_info


def test_standardize_with_support():
    """use random data to validate whether standardization process has data leakage"""
   support = np.random.randn(8, FEATURE_DIM)
    query = np.random.randn(4, FEATURE_DIM) + 5
    sx, qx, _ = standardize_with_support(support, query)
    scaler_s = StandardScaler().fit(support)
    scaler_all = StandardScaler().fit(np.vstack([support, query]))
    assert not (
            np.allclose(scaler_s.mean_, scaler_all.mean_) and
            np.allclose(scaler_s.scale_, scaler_all.scale_)
    ), "query set participated in fittingdata leakage"
    print("Standardization process unit test passed")


def diagnose_standardization(func):
    """decorator: after task samplingautomatically diagnosehealth of standardization process"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        """execute decorated function and outputdiagnostic infoonly return training processrequired standardization results"""

        result = func(*args, **kwargs)
        if not isinstance(result, (tuple, list)) or len(result) < 7:
            raise valueError("diagnostic function must returnresults tuple containing original features")

        sx, sy, qx, qy, info,support_x_raw, query_x_raw = result

        def _ensure_numpy(array_like):
            """safely convert tensor or arrayto NumPy arrayfor statistical analysis"""

            if isinstance(array_like, torch.Tensor):
                return array_like.detach().cpu().numpy()
            return np.asarray(array_like)

        # check 1:support setafter standardization mean value and standard deviation
        sx_np = _ensure_numpy(sx)
        if sx_np.size > 0:
            col_means = sx_np.mean(axis=0)
            col_stds = sx_np.std(axis=0)
            print(
                "[Diagnostic] Support set column mean value: "
                f"{float(col_means.mean()):.6f}, Support set column standard deviation mean value: {float(col_stds.mean()):.6f}"
            )
        else:
            print("[Diagnostic] Support set is empty, cannot calculate standardization statistics.")

        # check 2: compare originalsupport set and query setdistribution centers
       support_raw_np = _ensure_numpy(support_x_raw)
        query_raw_np = _ensure_numpy(query_x_raw)
        ifsupport_raw_np.size > 0 and query_raw_np.size > 0:
            scaler_s = StandardScaler().fit(support_raw_np)
            scaler_q = StandardScaler().fit(query_raw_np)
            if np.allclose(scaler_s.mean_, scaler_q.mean_):
                print("SEVERE WARNING:support set and query set distributions may be homogeneous orleakage risk exists")
        else:
            print("[Diagnostic] Insufficient original features, skip leakage risk diagnosis.")

        # check 3: note zero variance columnsre-insertion cases
        zero_var_cols = []
        if isinstance(info, dict):
            zero_var_cols = info.get("zero_var_cols_reinserted", []) or []
        if zero_var_cols:
            print(f"[Diagnosis] current task  contains re-inserted zero variance columns: {zero_var_cols}")

        return sx, sy, qx, qy, info

    return wrapper


# ===== Training/Validation/Test setConstruct =====
def split_task_dict(task_dict, train_ratio=0.6, val_ratio=0.2, leave_sensors=None):
    """based on each family internal ratio random split tasks dictionary."""
    if leave_sensors is not None:
        # leave specific sensor scenariosstill use original splitting logic
        train = {
            k: v
            for k, v in task_dict.items()
            if not any(sid in leave_sensors for (_, sid) in v)
        }
        test = {
            k: v
            for k, v in task_dict.items()
            if any(sid in leave_sensors for (_, sid) in v)
        }
        val = {}
        return train, val, test

    train_tasks = {}
    val_tasks = {}
    test_tasks = {}

    # 1. first split tasks by familyinto groups
    fam2tasks = defaultdict(list)
    for task_key, task_data in task_dict.items():
        if len(task_data) < (K_SHOT + Q_SHOT):
            # insufficient samples to complete K+Q tasks, skip directly
            continue
        family = task_key.split("__")[0]
        fam2tasks[family].append(task_key)

    print("\n=== family task distribution (before ratio-based splitting) ===")
    for fam, tasks in fam2tasks.items():
        task_count = len(tasks)
        sample_count = sum(len(task_dict[t]) for t in tasks)
        print(f"Family [{fam}]: {task_count}Task, {sample_count}Sample")

    # 2. within each familysplit by ratio
    for family, tasks_list in fam2tasks.items():
        if not tasks_list:
            # if no available within familytasks, skip directly
            continue

        random.shuffle(tasks_list)

        n_total = len(tasks_list)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        # when very few samples, try best toguarantee training set andvalidation set have at least one task
        if n_total > 2 and n_train == 0:
            n_train = 1
        if n_total > 2 and n_val == 0:
            n_val = 1

        train_keys = tasks_list[:n_train]
        val_keys = tasks_list[n_train: n_train + n_val]
        test_keys = tasks_list[n_train + n_val:]

        for key in train_keys:
            if key in task_dict:
                train_tasks[key] = task_dict[key]
        for key in val_keys:
            if key in task_dict:
                val_tasks[key] = task_dict[key]
        for key in test_keys:
            if key in task_dict:
                test_tasks[key] = task_dict[key]

    print("\nsplitting results (by ratio):")
    train_tasks_count = len(train_tasks)
    val_tasks_count = len(val_tasks)
    test_tasks_count = len(test_tasks)
    train_samples_count = sum(len(v) for v in train_tasks.values())
    val_samples_count = sum(len(v) for v in val_tasks.values())
    test_samples_count = sum(len(v) for v in test_tasks.values())

    print(f"Training: {train_tasks_count}Task, {train_samples_count}Sample")
    print(f"Val: {val_tasks_count}Task, {val_samples_count}Sample")
    print(f"Test: {test_tasks_count}Task, {test_samples_count}Sample")

    return train_tasks, val_tasks, test_tasks


# ===== Normal Pool Splitting (Random) =====
def split_normal_pool(normal_pool, train_ratio=0.7):
    n = len(normal_pool)
    idx = list(range(n))
    random.shuffle(idx)
    n_train = int(n * train_ratio)
    train = [normal_pool[i] for i in idx[:n_train]]
    remain = [normal_pool[i] for i in idx[n_train:]]
    return train, remain


# ===== Pre-process Normal Pools (Filtering) =====
def _prefilter_normal_pool(normal_pool, expected_len=FEATURE_DIM):
    kept, drop_len, drop_nan = [], 0, 0
    for entry in normal_pool:
        if isinstance(entry, dict):
            vec = entry.get("features")
        else:
            vec = entry
            entry = {"features": entry}

        if vec is None or len(vec) != expected_len:
            drop_len += 1
            continue

        arr = np.asarray(vec, dtype=float)
        if not np.isfinite(arr).all():
            drop_nan += 1
            continue

        cleaned = dict(entry)
        cleaned["features"] = arr
        kept.append(cleaned)
    return kept, drop_len, drop_nan


# ===== cache utility functions =====
def _file_md5(path: str) -> str:
    """calculate file MD5 to decidewhether source file changed"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_or_build_normal_pool(normal_df, source_path: str):
    """read or build normal poolsand cache baseline variance info"""
    if USE_CACHE and os.path.exists(NORMAL_POOL_CACHE):
        with open(NORMAL_POOL_CACHE, "rb") as f:
            data = pickle.load(f)
        if data.get("md5") == _file_md5(source_path) and data.get("version") == 2:
            print("UseCacheofnormal pools")
            summary = data.get("baseline_stats")
            if summary:
                _print_baseline_summary(summary)
            return data["pool"]
        else:
            print("cache structure outdatedrebuild normal pools……")
    if not USE_CACHE:
        print(" currentDisabledCacherebuild normal pools……")

    tracker = BaselineStatsTracker("normal samples")
    pool = build_normal_pool(normal_df, tracker=tracker)
    summary = tracker.summary()
    _print_baseline_summary(summary)
    with open(NORMAL_POOL_CACHE, "wb") as f:
        pickle.dump({
            "md5": _file_md5(source_path),
            "pool": pool,
            "baseline_stats": summary,
            "version": 2,
        }, f)
    return pool


def load_or_build_task_dict(incident_df, source_path: str):
    """read or build exception task dictionary and cache baseline variance info"""
    if USE_CACHE and os.path.exists(TASK_DICT_CACHE):
        with open(TASK_DICT_CACHE, "rb") as f:
            data = pickle.load(f)
        if data.get("md5") == _file_md5(source_path):
            print("UseCacheofexceptiontask dictionary")
            summary = data.get("baseline_stats")
            if summary:
                _print_baseline_summary(summary)
            return data["task_dict"]
    if not USE_CACHE:
        print(" currentDisabledCacherebuildexceptiontask dictionary……")

    tracker = BaselineStatsTracker("exceptionSample")
    task_dict = build_abnormal_tasks(incident_df, tracker)
    summary = tracker.summary()
    _print_baseline_summary(summary)
    with open(TASK_DICT_CACHE, "wb") as f:
        pickle.dump({
            "md5": _file_md5(source_path),
            "task_dict": task_dict,
            "baseline_stats": summary
        }, f)
    return task_dict


def _hash_train_task_structure(train_tasks: Dict[str, List[tuple]]) -> str:
    """calculate training task structurehash value to determinewhether to rebuild resamplingCache"""
    digest = hashlib.md5()
    digest.update(str(len(train_tasks)).encode("utf-8"))
    for task_key in sorted(train_tasks.keys()):
        digest.update(task_key.encode("utf-8"))
        pairs = train_tasks[task_key]
        digest.update(str(len(pairs)).encode("utf-8"))
        sorted_pairs = sorted(
            (tuple(pair) if isinstance(pair, (list, tuple)) else (pair,) for pair in pairs)
        )
        for pair in sorted_pairs:
            for value in pair[:2]:
                digest.update(str(value).encode("utf-8"))
    return digest.hexdigest()


def _hash_normal_pool_structure(normal_entries: List[Dict[str, object]]) -> str:
    """calculate training phase normal sample poolfeature hash to detectwhether cache needs invalidation"""
    digest = hashlib.md5()
    digest.update(str(len(normal_entries)).encode("utf-8"))
    for entry in normal_entries:
        vec = entry.get("features") if isinstance(entry, dict) else None
        if vec is None:
            digest.update(b"no_features")
        else:
            arr = np.asarray(vec, dtype=np.float32)
            digest.update(arr.tobytes())
        for key in ("IncidentId", "SensorNumber", "Week", "tag"):
            if isinstance(entry, dict) and key in entry and entry[key] is not None:
                digest.update(str(entry[key]).encode("utf-8"))
    return digest.hexdigest()


def _collect_resample_cache_meta(
        train_tasks: Dict[str, List[tuple]],
        normal_train: List[Dict[str, object]],
        incident_source_path: optional[str],
        normal_source_path: optional[str],
) -> Dict[str, object]:
    """generate resampling cacheinfo to decide whethercache is still valid"""
    meta: Dict[str, object] = {
        "version": 1,
        "task_signature": _hash_train_task_structure(train_tasks),
        "normal_signature": _hash_normal_pool_structure(normal_train),
        "task_count": len(train_tasks),
        "normal_count": len(normal_train),
        "seed": SEED,
        "feature_dim": FEATURE_DIM,
        "k_shot": K_SHOT,
        "q_shot": Q_SHOT,
    }

    if incident_source_path and os.path.exists(incident_source_path):
        meta["incident_md5"] = _file_md5(incident_source_path)
    if normal_source_path and os.path.exists(normal_source_path):
        meta["normal_md5"] = _file_md5(normal_source_path)

    return meta


def _unwrap_sample_task(result):
    """parse task sampling return resultscompatible with old version5-tuple with new version7-tuple structure"""

    if not isinstance(result, (tuple, list)):
        raise TypeError("sample_task return results type exceptionexpected tuple or list")

    if len(result) == 5:
        sx, sy, qx, qy, info = result
        return sx, sy, qx, qy, info

    if len(result) >= 7:
        sx, sy, qx, qy, info = result[:5]
       support_x_raw, query_x_raw = result[5], result[6]
        if isinstance(info, dict):
            info = dict(info)
            info.setdefault("support_x_raw",support_x_raw)
            info.setdefault("query_x_raw", query_x_raw)
        return sx, sy, qx, qy, info

    raise valueError("sample_task returned results insufficient elementscannot unpack")


def sample_task(task_dict, normal_pool, *,
                k_shot=K_SHOT, q_shot=Q_SHOT, max_retry=50,
                mode="one_vs_rest",
                incident_df_all=None,
                expected_len=FEATURE_DIM,
                hard_neg: bool = False,  #  enable hard negative sample mininginvalid in oc_maml mode
                hard_strategy: str = "distance",  # "distance" | "loss"
                neg_oversample: int = 5,  # candidate negative sample quantityfactor
                model=None,  # hard_strategy="loss" use
                device="cpu",
                debug=False,
                other_abnormal_ratio=0.5,  # Newly added: ratio of other anomaly samples in negative sampleso
                current_split_tasks=None  # Newly added: all task dictionary for current dataset
                ):
    """
    Support SetSupport set
        purpose: in each task loop(inner loop) to performfast adaptation
        Timing: each episode during training process
        purpose: let models fastlearn current taskspecific patterns
    Query Setquery set
        purpose: in the same task totest adaptation effectscalculate meta-lossupdate outer loopouter loop
        timing: immediately aftersupport set adaptationstill in training process
        Usage: evaluate model generalization capability on current task for updating meta-parameters

    Taskdistinguish positive class and anomaliesi.e. identify anomaliessamples of a certain class versus normal samples + OtherClassexceptionSample

    Positive class: specific type of anomaly samples
        data source: from task_dict extracted exception samples
        Quantity: Each task has K_SHOT + Q_SHOT = 5 + 10 = 15 positive samples
           support set: 5 positive samples(for adaptation)
            query set: 10 positive samplesForTest
        Featurefeature dimensions for FEATURE_DIMsensor count × 25 time points of difference sequence

    Negative class: normal samples + other type anomaly samples
        normal samplesfrom normal_poolnormal traffic datarandomly sampled from
        other type exception samplesfrom other tasks' exception eventsrandomly sampled from
        mixing ratio: controlled by other_abnormal_ratio=0.5 controlsi.e. 50% other exceptions + 50% normal samples
        negative class quantity allocationwill differ based on mode
            one_vs_restMode
                Support Set5 negative samples
                Query Set10 negative samples
                total: 15 negative samples
            oc_mamlMode
               support set: 0 negative samplesonly use positive class for adaptation
                Query Set10 negative samples
                total: 10 negative samples

    EachTaskin
    one_vs_restMode
       support set: 5 positive samples (10 samples of a certain class) + 5 negative samples (other anomaly samples: samples: 5 × 0.5 = 2.5 → 2 (towards rounding down) normal samples5 - 2 = 3 → 3) [Foradaptation]
        query set: 10 positive samples (10 samples of a certain class) + 10 negative samples (other anomaly samples: samples: 10 × 0.5 = 5 → 5 (towards rounding down) normal samples10 - 5 = 5 → 5) [ForTest]
    oc_mamlMode
       support set: 5 positive samples(10 samples of a certain class) [only for adaptation]
        query set: 10 positive samples(10 samples of a certain class) + 10 negative samples(similarly: 5 normalamples + 5OtherClassexceptionSample) [ForTest]
    """
    assert incident_df_all is not None, "sample_task necessary incident_df_all"
    assert current_split_tasks is not None, "sample_task requires current_split_tasks (current dataset'stask dictionary"

    need_pos = k_shot + q_shot
    # oc_maml does not need negativesupportonly needs negative class in query
    need_neg = (k_shot + q_shot) if mode == "one_vs_rest" else q_shot

    keys = list(task_dict.keys())
    if not keys:
        raise RuntimeError("sample_task: task_dict is empty")

    def family_of(task_key):
        """fromTaskKeysExtractfamily"""
        return task_key.split("__")[0]

    def week_tag_of(task_key):
        """fromTaskKeysExtractweekAndtag"""
        parts = task_key.split("__")
        if len(parts) >= 3:
            return parts[1], parts[2]  # Week, tag
        return None, None

    def _entry_identifier(entry):
        """generate unique sample identifiersfacilitate deduplication"""
        if isinstance(entry, dict):
            uid = entry.get("uid")
            if uid is not None:
                return uid
            inc = entry.get("IncidentId")
            sen = entry.get("SensorNumber")
            task_name = entry.get("task")
            if inc is not None and sen is not None:
                return (inc, sen, task_name)
            return id(entry)
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            return tuple(entry)
        return id(entry)

    def _fetch_feature(entry):
        """Extract feature vectors based on input items"""
        if isinstance(entry, dict) and "features" in entry:
            arr = np.asarray(entry["features"], dtype=float)
            if arr.shape[0] != expected_len or not np.isfinite(arr).all():
                return None
            return arr.copy()
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            iid, sid = entry
            f = extract_abnormal_features(incident_df_all, iid, sid)
            if f is None or f.shape[0] != expected_len or not np.isfinite(f).all():
                return None
            return f.astype(float)
        return None

    for attempt in range(1, max_retry + 1):
        task = random.choice(keys)
        current_family = family_of(task)
        pos_pairs_all = task_dict[task]

        if len(pos_pairs_all) < need_pos:
            if debug: print(f"[sample_task] #{attempt} Task[{task}]positive class candidates only {len(pos_pairs_all)} < {need_pos}")
            continue

        # ---- positive class: prioritize selectingsame family, different week/tagsamples ----
        # first try from same familyfrom different tasks to collectSample
        same_family_tasks = [t for t in task_dict.keys() if family_of(t) == current_family and t != task]

        pos_feats_clean = []
        bad_pos = 0

        # first from current taskcollect some samples
        current_task_pairs = list(pos_pairs_all)
        random.shuffle(current_task_pairs)

        # take a portion from current taskSampleensure not all fromsame week/tag
        max_from_current = max(1, need_pos // 2)  # at least 1, at most half

        for entry in current_task_pairs:
            if len(pos_feats_clean) >= max_from_current:
                break
            f = _fetch_feature(entry)
            if f is None:
                bad_pos += 1
                continue
            pos_feats_clean.append(f)

        # then from same family's other tasksto collect samples
        if len(pos_feats_clean) < need_pos and same_family_tasks:
            for other_task in same_family_tasks:
                if len(pos_feats_clean) >= need_pos:
                    break
                other_pairs = task_dict[other_task]
                random.shuffle(other_pairs)

                for entry in other_pairs:
                    if len(pos_feats_clean) >= need_pos:
                        break
                    f = _fetch_feature(entry)
                    if f is None:
                        bad_pos += 1
                        continue
                    pos_feats_clean.append(f)

        # if still not enough, supplement from current task
        if len(pos_feats_clean) < need_pos:
            remaining_pairs = current_task_pairs[max_from_current:]
            for entry in remaining_pairs:
                if len(pos_feats_clean) >= need_pos:
                    break
                f = _fetch_feature(entry)
                if f is None:
                    bad_pos += 1
                    continue
                pos_feats_clean.append(f)

        if len(pos_feats_clean) < need_pos:
            if debug: print(f"[sample_task] #{attempt}  after cleaning positive class {len(pos_feats_clean)}/{need_pos} (bad: {bad_pos})")
            continue

        pos_support = pos_feats_clean[:k_shot]
        pos_query = pos_feats_clean[k_shot:k_shot + q_shot]

        # need_neg in oc_maml mode is q_shot; in normal mode is k_shot+q_shot
        if mode == "oc_maml":
            need_neg = q_shot
        else:
            need_neg = k_shot + q_shot

        # ---- negative class: mix normal dataand other exception dataonly from current dataset ----
        # Calculatenecessaryofnormal samplesAndother exceptionsSamplequantity
        need_other_abnormal = int(need_neg * other_abnormal_ratio)
        need_normal = need_neg - need_other_abnormal

        # collect other exception samplesonly select from current dataset
        other_abnormal_clean = []
        if need_other_abnormal > 0:
            # Get tasks from current dataset excluding other anomaly class tasks of current family
            other_tasks = [k for k in current_split_tasks.keys()
                           if family_of(k) != current_family]

            if debug:
                print(f"[sample_task] currentfamily: {current_family}")
                print(f"[sample_task] optionalOtherfamily: {[family_of(t) for t in other_tasks]}")

            if other_tasks:
                # sample from different classes as much as possibleincrease diversity
                want_other = max(need_other_abnormal * (neg_oversample if hard_neg else 1), need_other_abnormal)
                sampled_ids = set()

                # Group other tasks by family
                other_families = {}
                for other_task in other_tasks:
                    fam = family_of(other_task)
                    if fam not in other_families:
                        other_families[fam] = []
                    other_families[fam].append(other_task)

                # round-robin from different familiessampling
                family_cycle = itertools.cycle(list(other_families.keys()))
                collected = 0
                attempts = 0
                max_attempts = want_other * 3  # prevent infinite loop

                while collected < want_other and attempts < max_attempts:
                    try:
                        other_family = next(family_cycle)
                        family_tasks = other_families[other_family]
                        other_task = random.choice(family_tasks)
                        other_pairs = current_split_tasks[other_task]

                        if other_pairs:
                            # randomly select one sample
                            entry = random.choice(other_pairs)
                            entry_id = _entry_identifier(entry)
                            if entry_id in sampled_ids:
                                attempts += 1
                                continue
                            f = _fetch_feature(entry)
                            if f is not None:
                                other_abnormal_clean.append(f)
                                sampled_ids.add(entry_id)
                                collected += 1
                        attempts += 1
                    except StopIteration:
                        break

                if debug and collected < need_other_abnormal:
                    print(f"[sample_task] insufficient other exception samples: {collected}/{need_other_abnormal}")

        # collect normal samples
        normal_clean = []
        if need_normal > 0 and len(normal_pool) >= need_normal:
            want_normal = max(need_normal * (neg_oversample if hard_neg else 1), need_normal)
            idx = list(range(len(normal_pool)))
            random.shuffle(idx)
            for i in idx:
                entry = normal_pool[i]
                vec = entry.get("features") if isinstance(entry, dict) else entry
                if vec is None or len(vec) != expected_len:
                    continue
                arr = np.asarray(vec, dtype=float)
                if not np.isfinite(arr).all():
                    continue
                normal_clean.append(arr)
                if len(normal_clean) >= want_normal:
                    break

        # Merge negative samples
        neg_clean = other_abnormal_clean + normal_clean

        if len(neg_clean) < need_neg:
            if debug:
                print(f"[sample_task] #{attempt} total negative class insufficient {len(neg_clean)}/{need_neg} "
                      f"(other exceptions:{len(other_abnormal_clean)}, normal:{len(normal_clean)})")
            continue

        # ---------- calculate difficulty and select need_neg  ----------
        if hard_neg and len(neg_clean) > need_neg:
            # use positive classsupport for standardization baseline
            pos_support_arr = np.array(pos_support, dtype=float)  # Shape: [k_shot, D]
            mu = pos_support_arr.mean(axis=0, keepdims=True)
            std = pos_support_arr.std(axis=0, keepdims=True)
            std[std < 1e-6] = 1.0  # prevent division by zero

            neg_arr = np.array(neg_clean, dtype=float)  # [M, D]
            # first based on positive class statisticsdo z-score, then score
            neg_z = (neg_arr - mu) / std

            if hard_strategy == "distance":
                # L2 distance to positive class prototypesmaller is harder
                # prototype in z-space is 0-vectorso equivalent to ||neg_z||
                scores = np.linalg.norm(neg_z, ord=2, axis=1)
                # Select minimal need_neg 
                sel_idx = np.argsort(scores)[:need_neg]

            elif hard_strategy == "loss":
                if model is None:
                    raise valueError("hard_strategy='loss' requires passing model")
                model.eval()
                with torch.no_grad():
                    # usesupport's mean/variance to standardize negatives to training space
                    # also pos/query will be similarlystandardized, keep consistency
                    neg_z_t = torch.tensor(neg_z, dtype=torch.float32, device=device)
                    # here set model to accept [N, D] -> logits for adjustment; if it isdifferent interface please adapt
                    logits = model(neg_z_t)  # [M, 1] or [M, 2]
                    # Binary classification: calculate "negative class (0)" score for each sample for BCE loss
                    if logits.shape[-1] == 1:
                        # sigmoid + BCE
                        probs = torch.sigmoid(logits.squeeze(-1))
                        y0 = torch.zeros_like(probs)
                        # avoid log(0)
                        eps = 1e-7
                        loss_per = -(y0 * torch.log(probs + eps) + (1 - y0) * torch.log(1 - probs + eps))
                    else:
                        # logits for 2-class: Use CEtarget=0
                        y0 = torch.zeros(logits.size(0), dtype=torch.long, device=device)
                        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
                        loss_per = loss_fn(logits, y0)
                    # higher loss is harder
                    sel_idx = torch.topk(loss_per, k=need_neg, largest=True).indices.cpu().numpy()
            else:
                raise ValueError("hard_strategy can only be 'distance' or 'loss'")

            neg_clean = [neg_clean[i] for i in sel_idx]
        else:
            # if not using hard sample miningor insufficient samples, randomly select
            random.shuffle(neg_clean)
            neg_clean = neg_clean[:need_neg]

        # ---- at this point, neg_clean already selected need_neg hard/normal negative samples ----

        # ---------- Groupsupport / query ----------
        if mode == "oc_maml":
           support_x = np.array(pos_support, dtype=float)
           support_y = np.array([1] * k_shot, dtype=int)
            # oc_maml query: positive q_shot + hard negative q_shot
            query_x = np.array(pos_query + neg_clean[:q_shot], dtype=float)
            query_y = np.array([1] * q_shot + [0] * q_shot, dtype=int)
        else:
            neg_support = neg_clean[:k_shot]
            neg_query = neg_clean[k_shot:k_shot + q_shot]
           support_x = np.array(pos_support + neg_support, dtype=float)
           support_y = np.array([1] * k_shot + [0] * k_shot, dtype=int)
            query_x = np.array(pos_query + neg_query, dtype=float)
            query_y = np.array([1] * q_shot + [0] * q_shot, dtype=int)

        # ---------- standardization (keep your original ----------
       support_x_raw =support_x.copy()
        query_x_raw = query_x.copy()
       support_x, query_x, norm_info = standardize_with_support(support_x, query_x)
        using_resampled = any(isinstance(item, dict) and "features" in item for item in pos_pairs_all)
        # record sampling infofacilitate subsequent analysis of sampling distribution
        info = {
            "task": task,
            "family": current_family,
            "attempt": attempt,
            "pos_support": int((support_y == 1).sum()),
            "neg_support": int((support_y == 0).sum()),
            "pos_query": int((query_y == 1).sum()),
            "neg_query": int((query_y == 0).sum()),
            "zero_var_cols_reinserted": norm_info.get("zero_var_cols_reinserted", []),
            "low_var_cols_perturbed": norm_info.get("low_var_cols_perturbed", []),
            "leak_warning": norm_info.get("leak_warning", False),
            "data_source": "resampled" if using_resampled else "raw",
        }

        return (
            torch.tensor(support_x, dtype=torch.float32),
            torch.tensor(support_y, dtype=torch.long),
            torch.tensor(query_x, dtype=torch.float32),
            torch.tensor(query_y, dtype=torch.long),
            info,
           support_x_raw,
            query_x_raw,
        )


# ===== Model Base Class =====
class BaseModel(nn.Module, ABC):
    def __init__(self, input_dim=FEATURE_DIM, dropout_rate=0.3):
        super().__init__()
        self.input_dim = input_dim
        # dropout_rate parameter in subclassto control random dropoutprobability
        self.dropout_rate = dropout_rate

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_l2_loss(self):
        """calculate model regularization loss(for B-SMALL, this is KL divergence loss"""
        pass

    def named_params_dict(self):
        """Returnnecessarygradientofparameterdictionary"""
        od = OrderedDict()
        for name, p in self.named_parameters():
            if p.requires_grad:
                od[name] = p
        return od

    @abstractmethod
    def forward_with_weights(self, x, weights):
        """use specified weight for forward pass"""
        pass


# ===== introducing B-SMALL model algorithm =====
# Sparse variational dropout (SVD) for fully-connected layers
class SparseVDLinear(nn.Module):
    """
    Implements Sparse Variational Dropout (SVD) for fully-connected layers
    This layer replaces standard nn.Linear and introduces Bayesian regularization for model sparsity.
    Weights are sampled from a Gaussian distribution with learnable mean (theta) and variance (sigma^2).

    reference paper:
    - Molchanov et al., 2017, "Variational Dropout Sparsifies Deep Neural Networks"
    - Madan and Prasad, 2021, "B-SMALL: A Bayesian Neural Network Approach..."
    """

    def __init__(self, in_features, out_features, bias=True):
        """
        Initialize sparse fully-connected layer.

        parameters:
            in_features (int): Input feature dimensions
            out_features (int): Output feature dimensions
            bias (bool): Whether to use bias (default: True)
        """
        super(SparseVDLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 1. Define learnable parameters
        # theta: mean of weights, this is model evaluation (when enabled) of "deterministic" weight.
        #        Initialized with Kaiming He initialization for stable training.
        self.theta = nn.parameter(torch.empty(out_features, in_features))

        # log_sigma2: log of weight variance. Learned in log form to ensure variance (sigma^2) is always positive,
        #             and facilitates optimization. A small initial value (-10) means low initial weight variance,
        #             close to a deterministic network, aiding early training stability.
        self.log_sigma2 = nn.parameter(torch.full((out_features, in_features), -10.0))

        # Bias term is a standard deterministic parameter.
        if bias:
            self.bias = nn.parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        # 2. Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize theta and bias parameters.
        """
        # Initialize theta using Kaiming Uniform initialization, a mature weight initialization method.
        nn.init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        if self.bias is not None:
            # Standard bias item initialization.
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.theta)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define forward pass logic.
        This method behaves differently based on whether model is in training mode (.train()) or evaluation mode (.eval())
        to execute different operations.

        parameters:
            x (torch.Tensor): Input tensor, shape is (..., in_features)

        Returns:
            torch.Tensor: Outputtensor
        """
        # Modedecision
        # self.training is nn.Module's built-inboolean value
        # model.train() sets it to True, model.eval() sets it to False.
        if self.training:
            # Traininging mode: apply Local Reparameterization Trick
            # This is the core trick from SVD paper, moves noise from weights to activations,
            # greatly reducing gradient variance for stable training.
            # see SVD paper equation (17).

            # 1. Calculate output mean (gamma)
            #    gamma = x * E[w] = x * theta
            gamma = F.linear(x, self.theta, None)  # bias added separately at the end

            # 2. Calculate output variance (delta)
            #    sigma^2 = exp(log_sigma2)
            #    delta = Var(x * w) = x^2 * Var(w) = x^2 * sigma^2
            sigma2 = torch.exp(self.log_sigma2)
            delta = F.linear(x.pow(2), sigma2, None)  # bias added separately at the end

            # For numerical stability, prevent delta from becoming negative or extremely small.
            delta = F.softplus(delta, beta=1, threshold=20)  # softplus ensures non-negative

            # 3. Inject Gaussian noise
            #    Sample noise epsilon from standard normal distribution N(0, 1).
            #    Output = gamma + sqrt(delta) * epsilon
            epsilon = torch.randn_like(gamma)
            output = gamma + torch.sqrt(delta) * epsilon

            if self.bias is not None:
                output += self.bias

            return output
        else:
            # evaluation mode: perform deterministicforward pass
            # during evaluation or inferencewe don't want randomnesstherefore only use weight mean value a.
            # this is equivalent to a standard nn.Linear layer
            return F.linear(x, self.theta, self.bias)

    @property
    def log_alpha(self) -> torch.Tensor:
        """
        calculate and return a sparsity metrickey metric log_alpha
        alpha defines the inverse of signal-to-noise ratioalpha = sigma^2 / theta^2
        when alpha is very largeit means noise is much larger than signalthe corresponding weight can be“pruning”
        we calculate its log formto improve numerical stability

        Returns:
            torch.Tensor: log_alpha tensor
        """
        # to avoid log(0) leading to NaNadd a tiny value epsilon to theta epsilon
        epsilon = 1e-8
        return self.log_sigma2 - 2 * torch.log(torch.abs(self.theta) + epsilon)


# calculate KL divergence loss of single layer
def calculate_kl_loss(module: nn.Module) -> torch.Tensor:
    """
    Calculate KL divergence loss for a single sparse layer (e.g. SparseVDLinear).
    This loss is B-SMALL's regularization objective to penalize model complexityty and drive sparsity

    this implementation precisely follows SVD paper equation (14)with numerical stability processing

    parameters:
        module (nn.Module): containing log_alpha attributesparse layer instance

    Returns:
        torch.Tensor: total KL divergence loss of this layer (a scalar)
    """
    # ensure the passed module isour defined sparse layer
    if not isinstance(module, SparseVDLinear):  # can later extend to or isinstance(module, SparseVDConv2d)
        return torch.tensor(0.0, device=module.theta.device)

    # from SVD paper equation (14) constant
    k1, k2, k3 = 0.63576, 1.87320, 1.48695

    # Getmoduleof log_alpha
    log_alpha = module.log_alpha

    # calculate core part of KL divergence
    # SVD paper equation (14) gives -D_KL approximation
    # -D_KL ≈ k1 * sigmoid(k2 + k3*log_a) - 0.5 * log(1 + 1/a) + C
    # paper suggests taking C = -k1making when a -> ∞-D_KL -> 0
    # ThereforeD_KL ≈ - (k1 * sigmoid(...) - 0.5 * log(1 + 1/a) - k1)

    # log(1 + 1/a) = log(1 + exp(-log_a))using log1p can improve numerical stability
    term1 = k1 * torch.sigmoid(k2 + k3 * log_alpha)
    term2 = -0.5 * torch.log1p(torch.exp(-log_alpha))

    # adding constant C=-k1then negate to get D_KL
    kl_divergence = -(term1 + term2 - k1)

    # return sum of KL divergence of all weights in layer
    return torch.sum(kl_divergence)


# ======== B-SMALLmodelsAlgorithms =========


# ===== StandardMLPmodels =====
class StandardMLP(BaseModel):
    """
    optimized version: multi-layer perceptron modelimprove calculation efficiency

    Optimization points:
    1. use nn.Sequential to simplify forward pass
    2. OptimizedropoutlayerofUse
    3. add batch normalization to improve training stability
    4. use more efficientactivationFunction
    """
    def __init__(self, input_dim=FEATURE_DIM, l2_reg=1e-4, dropout_rate=0.4):
        super().__init__(input_dim, dropout_rate)
        self.l2_reg = l2_reg

        # OptimizeUsenn.ModuleDictfacilitateatforward_with_weightsinAccess
        self.layers = nn.ModuleDict({
            'fc1': nn.Linear(input_dim, 256),
            'bn1': nn.BatchNorm1d(256),  # Add batch normalization
            'fc2': nn.Linear(256, 256),
            'bn2': nn.BatchNorm1d(256),
            'fc3': nn.Linear(256, 128),
            'bn3': nn.BatchNorm1d(128),
            'fc4': nn.Linear(128, 64),
            'bn4': nn.BatchNorm1d(64),
            'fc5': nn.Linear(64, 2)
        })

        # Optimization: use only one dropout layer at key locations
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Optimization: use more concise forward pass, reduce variables
        for i in range(1, 5):  # fc1 to fc4
            x = self.layers[f'fc{i}'](x)

            # batch normalization (only during training(when enabled)
            if hasattr(self, f'bn{i}') and self.training:
                x = self.layers[f'bn{i}'](x)

            x = F.gelu(x)  # optimization: use GELU instead of ReLUsmoother

            # Optimization: only apply dropout between layers, not before final layer
            if i < 4:
                x = self.dropout(x)

        # Output layer (no activation function)
        return self.layers['fc5'](x)

    def get_l2_loss(self):
        # Optimizeuse more efficient L2 regularization calculation
        l2_loss = sum(param.norm() ** 2 for param in self.parameters())
        return self.l2_reg * l2_loss

    def forward_with_weights(self, x, weights):
        """optimized versionSupportweightparameterofforward pass"""
        # Optimizedirectly use functional APIavoid repeated dropout logic
        x = F.linear(x, weights['fc1.weight'], weights.get('fc1.bias'))
        if self.training and 'bn1.weight' in weights:
            x = F.batch_norm(x, running_mean=None, running_var=None,
                           weight=weights.get('bn1.weight'), bias=weights.get('bn1.bias'),
                           training=True, momentum=0.1)
        x = F.gelu(x)
        if self.training:
            x = F.dropout(x, p=self.dropout_rate)

        x = F.linear(x, weights['fc2.weight'], weights.get('fc2.bias'))
        if self.training and 'bn2.weight' in weights:
            x = F.batch_norm(x, running_mean=None, running_var=None,
                           weight=weights.get('bn2.weight'), bias=weights.get('bn2.bias'),
                           training=True, momentum=0.1)
        x = F.gelu(x)
        if self.training:
            x = F.dropout(x, p=self.dropout_rate)

        x = F.linear(x, weights['fc3.weight'], weights.get('fc3.bias'))
        if self.training and 'bn3.weight' in weights:
            x = F.batch_norm(x, running_mean=None, running_var=None,
                           weight=weights.get('bn3.weight'), bias=weights.get('bn3.bias'),
                           training=True, momentum=0.1)
        x = F.gelu(x)
        if self.training:
            x = F.dropout(x, p=self.dropout_rate)

        x = F.linear(x, weights['fc4.weight'], weights.get('fc4.bias'))
        if self.training and 'bn4.weight' in weights:
            x = F.batch_norm(x, running_mean=None, running_var=None,
                           weight=weights.get('bn4.weight'), bias=weights.get('bn4.bias'),
                           training=True, momentum=0.1)
        x = F.gelu(x)
        # Do not apply dropout before output layer

        return F.linear(x, weights['fc5.weight'], weights.get('fc5.bias'))


# ===== 1D-CNNmodels =====
class CNN1D(BaseModel):
    def __init__(self, input_dim=FEATURE_DIM, dropout_rate=0.3, l2_reg=1e-4):
        super().__init__(input_dim, dropout_rate)
        self.l2_reg = l2_reg
        self.num_channels = FEATURE_CHANNELS
        self.seq_len = 25
        # reshape to [batch, channels, 25] for 1D convolution
        self.conv1 = nn.Conv1d(self.num_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        # x: [batch, FEATURE_DIM] -> [batch, num_channels, 25]
        x = x.view(-1, self.num_channels, self.seq_len)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)  # [batch, 64]
        x = self.dropout(x)
        return self.fc(x)

    def get_l2_loss(self):
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param, 2) ** 2
        return self.l2_reg * l2_loss

    def forward_with_weights(self, x, weights):
        # Reshape input
        x = x.view(-1, self.num_channels, self.seq_len)

        # Conv1
        w1, b1 = weights["conv1.weight"], weights["conv1.bias"]
        x = F.conv1d(x, w1, b1, padding=1)
        x = F.relu(x)

        # Conv2
        w2, b2 = weights["conv2.weight"], weights["conv2.bias"]
        x = F.conv1d(x, w2, b2, padding=1)
        x = F.relu(x)

        # Conv3
        w3, b3 = weights["conv3.weight"], weights["conv3.bias"]
        x = F.conv1d(x, w3, b3, padding=1)
        x = F.relu(x)

        # Pool and FC
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        wfc, bfc = weights["fc.weight"], weights["fc.bias"]
        x = F.linear(x, wfc, bfc)
        return x


# ===== LSTMmodels =====
class LSTM(BaseModel):
    def __init__(self, input_dim=FEATURE_DIM, dropout_rate=0.3, l2_reg=1e-4, hidden_dim=64, num_layers=2):
        super().__init__(input_dim, dropout_rate)
        self.l2_reg = l2_reg
        self.input_channels = FEATURE_CHANNELS
        self.seq_len = 25
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # manually implement LSTM layertosupport forward_with_weights
        self.lstm_layers = nn.ModuleList([
            LSTMLayer(self.input_channels if i == 0 else hidden_dim, hidden_dim, dropout_rate)
            for i in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        # x: [batch, FEATURE_DIM] -> [batch, 25, num_channels]
        x = x.view(-1, self.seq_len, self.input_channels)
        batch_size, seq_len, _ = x.shape

        # Initializehiddenstatus
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        c = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        # ThroughEach LSTM layer
        for layer in self.lstm_layers:
            h, c = layer(x, h, c)
            x = h.unsqueeze(1).expand(-1, seq_len, -1)  # prepare input for next layer

        # UseLastOnewhenstepofOutput
        x = h  # [batch, hidden_dim]
        x = self.dropout(x)
        return self.fc(x)

    def get_l2_loss(self):
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param, 2) ** 2
        return self.l2_reg * l2_loss

    def forward_with_weights(self, x, weights):
        # Reshape input
        x = x.view(-1, self.seq_len, self.input_channels)
        batch_size, seq_len, _ = x.shape

        # Initializehiddenstatus
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        c = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        # ThroughEach LSTM layer
        for layer_idx in range(self.num_layers):
            h, c = self._lstm_layer_forward_with_weights(x, h, c, weights, layer_idx)
            if layer_idx < self.num_layers - 1:  # not the last layer
                x = h.unsqueeze(1).expand(-1, seq_len, -1)  # prepare input for next layer

        # FC layer
        wfc = weights["fc.weight"]
        bfc = weights["fc.bias"]
        x = F.linear(h, wfc, bfc)
        return x

    def _lstm_layer_forward_with_weights(self, x, h_prev, c_prev, weights, layer_idx):
        """SingleLSTMlayerofforward pass"""
        prefix = f"lstm_layers.{layer_idx}."
        seq_len = x.size(1)

        # GetLSTMweight
        w_ih = weights[f"{prefix}weight_ih"]
        w_hh = weights[f"{prefix}weight_hh"]
        b_ih = weights[f"{prefix}bias_ih"]
        b_hh = weights[f"{prefix}bias_hh"]

        h, c = h_prev, c_prev

        # Calculate step by step
        for t in range(seq_len):
            h, c = self._lstm_cell_forward(x[:, t], h, c, w_ih, w_hh, b_ih, b_hh)

        return h, c

    def _lstm_cell_forward(self, input_t, h_prev, c_prev, w_ih, w_hh, b_ih, b_hh):
        """LSTMunitofforward pass"""
        # calculate gates
        gi = F.linear(input_t, w_ih, b_ih)
        gh = F.linear(h_prev, w_hh, b_hh)
        i_i, i_f, i_g, i_o = gi.chunk(4, 1)
        h_i, h_f, h_g, h_o = gh.chunk(4, 1)

        inputgate = torch.sigmoid(i_i + h_i)
        forgetgate = torch.sigmoid(i_f + h_f)
        cellgate = torch.tanh(i_g + h_g)
        outputgate = torch.sigmoid(i_o + h_o)

        # update cell state and hidden state
        c_new = forgetgate * c_prev + inputgate * cellgate
        h_new = outputgate * torch.tanh(c_new)

        return h_new, c_new


class LSTMLayer(nn.Module):
    """Custom LSTM layer compatible with MAML"""

    def __init__(self, input_size, hidden_size, dropout_rate=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # LSTMweight
        self.weight_ih = nn.parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.parameter(torch.randn(4 * hidden_size))

        self.dropout = nn.Dropout(dropout_rate)
        self.init_weights()

    def init_weights(self):
        """Initializeweight"""
        std = 1.0 / (self.hidden_size) ** 0.5
        for weight in [self.weight_ih, self.weight_hh]:
            weight.data.uniform_(-std, std)
        self.bias_ih.data.fill_(0)
        self.bias_hh.data.fill_(0)
        # forget gate bias set to 1common practice
        forget_gate_start = self.hidden_size
        forget_gate_end = 2 * self.hidden_size
        self.bias_ih.data[forget_gate_start:forget_gate_end].fill_(1)
        self.bias_hh.data[forget_gate_start:forget_gate_end].fill_(1)

    def forward(self, x, h_0, c_0):
        """
        Args:
            x: [batch, seq_len, input_size]
            h_0: [batch, hidden_size]
            c_0: [batch, hidden_size]
        Returns:
            h_n: [batch, hidden_size] - hidden state at last time step
            c_n: [batch, hidden_size] - final cell state
        """
        batch_size, seq_len, _ = x.shape
        h, c = h_0, c_0

        for t in range(seq_len):
            h, c = self._lstm_cell(x[:, t], h, c)

        return h, c

    def _lstm_cell(self, input_t, h_prev, c_prev):
        """LSTMunitCalculate"""
        # input gate, forget gate, candidate value, output gate linear transformations
        gi = F.linear(input_t, self.weight_ih, self.bias_ih)
        gh = F.linear(h_prev, self.weight_hh, self.bias_hh)
        i_i, i_f, i_g, i_o = gi.chunk(4, 1)
        h_i, h_f, h_g, h_o = gh.chunk(4, 1)

        # calculate gate values
        inputgate = torch.sigmoid(i_i + h_i)
        forgetgate = torch.sigmoid(i_f + h_f)
        cellgate = torch.tanh(i_g + h_g)
        outputgate = torch.sigmoid(i_o + h_o)

        # update cell state and hidden state
        c_new = forgetgate * c_prev + inputgate * cellgate
        h_new = outputgate * torch.tanh(c_new)

        return h_new, c_new


# ===== position encoding module =====
class PositionalEncoding(nn.Module):
    """Standard Transformer uses sinusoidal position encoding,implementing sequence location information injection"""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # register buffer for saving with modelwhile avoiding parameter gradientCalculate
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """add pre-calculated position encoding to input tensors"""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


# ===== StandardTransformermodels =====
def resolve_activation_fn(name: str) -> nn.Module:
    """generate corresponding based on nameactivation function instanceguarantee configurability and readability"""

    name_lower = name.lower()
    activation_map = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "elu": nn.ELU,
    }
    if name_lower not in activation_map:
        raise valueError("unsupported activation function typeplease select from relu/gelu/silu/elu")
    return activation_map[name_lower]()


class MultiHeadSelfAttention(nn.Module):
    """manually implemented multi-head attentionusing basis matrix operations to avoid high-order gradient issues"""

    def __init__(self, d_model: int, nhead: int, dropout_rate: float) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise valueError("d_model must be divisible by nheadto evenly divide each attention head's dimensions")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scaling = self.head_dim ** -0.5

        # use independent linear layers to generate QueryKey, Valuefacilitate reading and debugging
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """calculate self-attention outputsupport second-order gradients for MAML training"""

        batch_size, seq_len, _ = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # ShapeConvertfor [batch, nhead, seq_len, head_dim]
        def reshape_heads(tensor: torch.Tensor) -> torch.Tensor:
            tensor = tensor.view(batch_size, seq_len, self.nhead, self.head_dim)
            return tensor.transpose(1, 2)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        context = torch.matmul(attn_weights, v)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(context)


class TransformerEncoderBlock(nn.Module):
    """adopt pre-normalization structure of Transformer encoder layercompatible with high-order gradients"""

    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int,
            dropout_rate: float,
            activation: str,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, nhead, dropout_rate)
        self.dropout_attn = nn.Dropout(dropout_rate)
        self.dropout_ffn = nn.Dropout(dropout_rate)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            resolve_activation_fn(activation),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """execute attention with feed-forward networkboth adopt residual connections"""

        # self-attention sublayeradopt pre-normalization to improve stability
        attn_input = self.norm_attn(x)
        attn_output = self.self_attn(attn_input)
        x = x + self.dropout_attn(attn_output)

        # feed-forward sublayersimilarly adopt pre-normalization with residual
        ffn_input = self.norm_ffn(x)
        ffn_output = self.ffn(ffn_input)
        x = x + self.dropout_ffn(ffn_output)
        return x


class StandardTransformer(BaseModel):
    """integrated self-defined attention layer ofStandard Transformer classifiersupport high-order gradients"""

    def __init__(
            self,
            input_dim: int = FEATURE_DIM,
            d_model: int = 128,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 256,
            dropout_rate: float = 0.3,
            l2_reg: float = 1e-4,
            activation: str = "gelu",
            sequence_length: optional[int] = None,
            feature_channels: optional[int] = None,
            **kwargs,
    ) -> None:
        super().__init__(input_dim, dropout_rate)
        # parse sequence length and channel countensureInputCanreshape to [batch, seq_len, feature_channels]
        resolved_channels = feature_channels if feature_channels is not None else FEATURE_CHANNELS
        if resolved_channels <= 0:
            raise valueError("feature_channels must be positive integerensure sequence dimension can be divided")
        if sequence_length is None:
            if input_dim % resolved_channels != 0:
                raise valueError(
                    "input_dim cannot be divided by feature_channelsplease explicitly pass sequence_length to match original sequence length"
                )
            sequence_length = input_dim // resolved_channels
        if sequence_length <= 0:
            raise ValueError("sequence_length must be > 0, ensure sequence contains valid time step")
        self.sequence_length = sequence_length
        self.feature_channels = resolved_channels

        self.l2_reg = l2_reg
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        # map each time step's features to Transformer's hidden dimension
        self.input_projection = nn.Linear(self.feature_channels, d_model)
        self.embedding_scale = math.sqrt(d_model)
        self.embed_dropout = nn.Dropout(dropout_rate)
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.sequence_length + 1)
        # add learnable CLS tokenfor aggregating global semantics
        self.cls_token = nn.parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # build self-defined encoder layer stackavoid adapting official implementationleading to high-order gradient issues
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout_rate=dropout_rate,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """execute forward passmap input features tobinary classification logits"""
        if x.dim() != 2:
            raise ValueError("Transformer input must be [batch, feature_dim], please ensure preprocessing output is correct")
        if x.size(1) != self.input_dim:
            raise valueError("input feature dimensions inconsistent with model initializationplease check feature concatenation process")

        # reshape to [batch, seq_len, feature_channels] then perform linear projection
        batch_size = x.size(0)
        x = x.view(batch_size, self.sequence_length, self.feature_channels)
        x = self.input_projection(x) * self.embedding_scale
        x = torch.cat([self.cls_token.expand(batch_size, -1, -1), x], dim=1)
        x = self.pos_encoder(x)
        x = self.embed_dropout(x)

        for encoder in self.encoder_layers:
            x = encoder(x)

        cls_state = x[:, 0]
        cls_state = self.final_norm(cls_state)
        cls_state = self.dropout(cls_state)
        return self.classifier(cls_state)

    def get_l2_loss(self) -> torch.Tensor:
        """Calculate L2 norm regularization of model parameters."""
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss = l2_loss + torch.norm(param, 2) ** 2
        return self.l2_reg * l2_loss

    def forward_with_weights(self, x: torch.Tensor, weights: OrderedDict) -> torch.Tensor:
        """Execute forward pass with specified weightsCompatible with MAML's fast adaptation steps"""
        # Uniformly use compatible version of functional_call (adapt to PyTorch 2.x API changes)
        if _functional_call_compat is None:
            raise RuntimeError("Current environment lacks functional_call (PyTorch version too old  or API location change, please upgrade to >=2.0")
        return _functional_call_compat(self, weights, (x,))


# ===== B-SMALL Transformer models =====
class B_SMALL_MultiHeadSelfAttention(nn.Module):
    """Use SparseVDLinear to replace nn.Linear in multi-head self-attention layer,supporting B-SMALL sparsification."""""

    def __init__(self, d_model: int, nhead: int, dropout_rate: float) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise valueError("d_model must be divisible by nheadto evenly divide each attention head's dimensions")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scaling = self.head_dim ** -0.5

        # Use SparseVDLinear ReplaceStandardlinearlayer
        self.q_proj = SparseVDLinear(d_model, d_model)
        self.k_proj = SparseVDLinear(d_model, d_model)
        self.v_proj = SparseVDLinear(d_model, d_model)
        self.out_proj = SparseVDLinear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """calculate self-attention outputsupport second-order gradients for MAML training"""

        batch_size, seq_len, _ = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # ShapeConvertfor [batch, nhead, seq_len, head_dim]
        def reshape_heads(tensor: torch.Tensor) -> torch.Tensor:
            tensor = tensor.view(batch_size, seq_len, self.nhead, self.head_dim)
            return tensor.transpose(1, 2)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        context = torch.matmul(attn_weights, v)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(context)


class B_SMALL_TransformerEncoderBlock(nn.Module):
    """Use SparseVDLinear in Transformer encoder layer,supporting B-SMALL sparsification."""

    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int,
            dropout_rate: float,
            activation: str,
    ) -> None:
        super().__init__()
        self.self_attn = B_SMALL_MultiHeadSelfAttention(d_model, nhead, dropout_rate)
        self.dropout_attn = nn.Dropout(dropout_rate)
        self.dropout_ffn = nn.Dropout(dropout_rate)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

        # FFN Use SparseVDLinear
        self.ffn_linear1 = SparseVDLinear(d_model, dim_feedforward)
        self.ffn_activation = resolve_activation_fn(activation)
        self.ffn_dropout = nn.Dropout(dropout_rate)
        self.ffn_linear2 = SparseVDLinear(dim_feedforward, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """execute attention with feed-forward networkboth adopt residual connections"""

        # self-attention sublayeradopt pre-normalization to improve stability
        attn_input = self.norm_attn(x)
        attn_output = self.self_attn(attn_input)
        x = x + self.dropout_attn(attn_output)

        # feed-forward sublayersimilarly adopt pre-normalization with residual
        ffn_input = self.norm_ffn(x)
        ffn_output = self.ffn_linear1(ffn_input)
        ffn_output = self.ffn_activation(ffn_output)
        ffn_output = self.ffn_dropout(ffn_output)
        ffn_output = self.ffn_linear2(ffn_output)
        x = x + self.dropout_ffn(ffn_output)
        return x


class B_SMALL_Transformer(BaseModel):
    """Use SparseVDLinear in B-SMALL Transformer classifier, implementing sparsity via KL  divergence."""

    def __init__(
            self,
            input_dim: int = FEATURE_DIM,
            d_model: int = 128,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 256,
            dropout_rate: float = 0.3,
            activation: str = "gelu",
            sequence_length: optional[int] = None,
            feature_channels: optional[int] = None,
            **kwargs,
    ) -> None:
        super().__init__(input_dim, dropout_rate)
        # parse sequence length and channel countensureInputCanreshape to [batch, seq_len, feature_channels]
        resolved_channels = feature_channels if feature_channels is not None else FEATURE_CHANNELS
        if resolved_channels <= 0:
            raise valueError("feature_channels must be positive integerensure sequence dimension can be divided")
        if sequence_length is None:
            if input_dim % resolved_channels != 0:
                raise valueError(
                    "input_dim cannot be divided by feature_channelsplease explicitly pass sequence_length to match original sequence length"
                )
            sequence_length = input_dim // resolved_channels
        if sequence_length <= 0:
            raise ValueError("sequence_length must be > 0, ensure sequence contains valid time step")
        self.sequence_length = sequence_length
        self.feature_channels = resolved_channels

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        # map each time step's features to Transformer's hidden dimension - Use SparseVDLinear
        self.input_projection = SparseVDLinear(self.feature_channels, d_model)
        self.embedding_scale = math.sqrt(d_model)
        self.embed_dropout = nn.Dropout(dropout_rate)
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.sequence_length + 1)
        # add learnable CLS tokenfor aggregating global semantics
        self.cls_token = nn.parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Build B-SMALL encoder layer stack
        self.encoder_layers = nn.ModuleList(
            [
                B_SMALL_TransformerEncoderBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout_rate=dropout_rate,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        # Classifier uses SparseVDLinear
        self.classifier = SparseVDLinear(d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """execute forward passmap input features tobinary classification logits"""
        if x.dim() != 2:
            raise ValueError("Transformer input must be [batch, feature_dim], please ensure preprocessing output is correct")
        if x.size(1) != self.input_dim:
            raise valueError("input feature dimensions inconsistent with model initializationplease check feature concatenation process")

        # reshape to [batch, seq_len, feature_channels] then perform linear projection
        batch_size = x.size(0)
        x = x.view(batch_size, self.sequence_length, self.feature_channels)
        x = self.input_projection(x) * self.embedding_scale
        x = torch.cat([self.cls_token.expand(batch_size, -1, -1), x], dim=1)
        x = self.pos_encoder(x)
        x = self.embed_dropout(x)

        for encoder in self.encoder_layers:
            x = encoder(x)

        cls_state = x[:, 0]
        cls_state = self.final_norm(cls_state)
        cls_state = self.dropout(cls_state)
        return self.classifier(cls_state)

    def get_l2_loss(self) -> torch.Tensor:
        """
        B-SMALL model uses KL divergence as regularization loss, not L2 regularization.
        To maintain interface compatibility, this method returns KL divergence loss.
        """
        return self.get_regularization_loss()

    def get_regularization_loss(self) -> torch.Tensor:
        """
        Implement regularization loss calculation for B-SMALL model, i.e., total KL divergence loss.
        Iterate through all modules, accumulate KL divergence of SparseVDLinear layers.
        """
        total_kl_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for module in self.modules():
            if isinstance(module, SparseVDLinear):
                total_kl_loss += calculate_kl_loss(module)
        return total_kl_loss

    def forward_with_weights(self, x: torch.Tensor, weights: OrderedDict) -> torch.Tensor:
        """Execute forward pass with specified weightsCompatible with MAML's fast adaptation steps"""
        if _functional_call_compat is None:
            raise RuntimeError("Current environment lacks functional_call (PyTorch version too old  or API location change, please upgrade to >=2.0")
        return _functional_call_compat(self, weights, (x,))


# ===== metrics logging and visualization tools =====
def save_metrics(train_log, val_log, out_dir="metrics"):
    """SaveTrainingwithValidationPhaseofmetricsfor CSV file"""
    if DISABLE_PLOTTING:
        # Only save CSV, do not perform any plotting
        os.makedirs(out_dir, exist_ok=True)
        if train_log:
            train_path = os.path.join(out_dir, "train_metrics.csv")
            pd.DataFrame(train_log).to_csv(train_path, index=False)
            print(f"Saved training metrics to {train_path}")
        if val_log:
            val_path = os.path.join(out_dir, "val_metrics.csv")
            pd.DataFrame(val_log).to_csv(val_path, index=False)
            print(f"Saved validation metrics to {val_path}")
        return
    os.makedirs(out_dir, exist_ok=True)
    if train_log:
        train_path = os.path.join(out_dir, "train_metrics.csv")
        pd.DataFrame(train_log).to_csv(train_path, index=False)
        print(f"Saved training metrics to {train_path}")
    if val_log:
        val_path = os.path.join(out_dir, "val_metrics.csv")
        pd.DataFrame(val_log).to_csv(val_path, index=False)
        print(f"Saved validation metrics to {val_path}")


def save_family_metrics(fam_log, out_path):
    """Savebased on family statisticsofmetrics"""
    if not fam_log:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(fam_log).to_csv(out_path, index=False)
    print(f"Saved family metrics to {out_path}")


def plot_metrics(train_log, val_log, out_dir="metrics"):
    """Plot training and validation metric curves and save as images"""
    if DISABLE_PLOTTING:
        print("[Plotting Disabled] Skipped plot_metrics")
        return
    if not train_log and not val_log:
        return
    os.makedirs(out_dir, exist_ok=True)
    train_df = pd.DataFrame(train_log)
    val_df = pd.DataFrame(val_log)
    metrics = ["meta_loss", "acc", "recall", "f1"]
    for m in metrics:
        plt.figure()
        if not train_df.empty:
            plt.plot(train_df["episode"], train_df[m], label="train")
        if not val_df.empty:
            plt.plot(val_df["episode"], val_df[m], label="val")
        plt.xlabel("Episode")
        plt.ylabel(m)
        plt.title(f"{m} Curve")
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{m}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Generated metrics curve {out_path}")


def save_family_summary(family_metrics_dict: Dict[str, Dict[str, float]], out_path: str) -> None:
    """Save performance overview for each family based on final evaluation results."""
    if not family_metrics_dict:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []
    for fam, stats in family_metrics_dict.items():
        row = {"family": fam}
        row.update(stats)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved family metrics overview to {out_path}")


def plot_eval_artifacts(probs: List[float], labels: List[int], preds: List[int], out_dir: str, prefix: str) -> None:
    """Plot PR/ROC/confusion matrix/probability histogram charts for evaluation phase."""
    if DISABLE_PLOTTING:
        print("[Plotting Disabled] Skipped evaluation figure plotting")
        return
    if not probs or not labels:
        print(f"{prefix} phase lacks probabilities or labels, skipped figure plotting")
        return

    os.makedirs(out_dir, exist_ok=True)
    probs_arr = np.asarray(probs, dtype=float)
    labels_arr = np.asarray(labels, dtype=int)
    preds_arr = np.asarray(preds, dtype=int)

    # ===== Plot PR Curve =====
    if np.unique(labels_arr).size > 1:
        precision, recall_vals, _ = precision_recall_curve(labels_arr, probs_arr)
        plt.figure()
        plt.plot(recall_vals, precision, label="PR Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{prefix.upper()} PR Curve")
        plt.tight_layout()
        pr_path = os.path.join(out_dir, f"pr_curve_{prefix}.png")
        plt.savefig(pr_path)
        plt.close()
        print(f"Saved {prefix} phase PR curve: {pr_path}")
    else:
        print(f"{prefix} phase contains only one class, cannot plot PR curve")

    # ===== Plot ROC Curve =====
    if np.unique(labels_arr).size > 1:
        fpr, tpr, _ = roc_curve(labels_arr, probs_arr)
        plt.figure()
        plt.plot(fpr, tpr, label="ROC Curve")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{prefix.upper()} ROC Curve")
        plt.tight_layout()
        roc_path = os.path.join(out_dir, f"roc_curve_{prefix}.png")
        plt.savefig(roc_path)
        plt.close()
        print(f"Saved {prefix} phase ROC curve: {roc_path}")
    else:
        print(f"{prefix} phase contains only one class, cannot plot ROC curve")

    # ===== Plot Confusion Matrix =====
    cm = confusion_matrix(labels_arr, preds_arr, labels=[0, 1])
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"{prefix.upper()} Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Negative", "Positive"])
    plt.yticks(tick_marks, ["Negative", "Positive"])
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, f"cm_{prefix}.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved {prefix} phase confusion matrix: {cm_path}")

    # ===== Plot Probability Histogram =====
    plt.figure()
    plt.hist(probs_arr, bins=20, range=(0, 1))
    plt.xlabel("Predicted Positive Probability")
    plt.ylabel("Count")
    plt.title(f"{prefix.upper()} Probability Histogram")
    plt.tight_layout()
    hist_path = os.path.join(out_dir, f"proba_hist_{prefix}.png")
    plt.savefig(hist_path)
    plt.close()
    print(f"Saved {prefix} phase probability histogram: {hist_path}")


# ===== AnalyzeTask samplinglog =====
def analyze_task_sampling_log(log_path="task_sampling_log.csv"):
    """Read and summarize task sampling logs, report sampling and failure counts for each task"""
    if not os.path.exists(log_path):
        print("Not found task_sampling_log.csvSkippedAnalyze")
        return
    df = pd.read_csv(log_path)
    success_df = df[df["status"] == "success"]
    fail_df = df[df["status"] == "failed"]
    success_cnt = success_df.groupby("task").size().rename("success_count")
    if "task" in fail_df.columns:
        fail_cnt = fail_df.groupby("task").size().rename("fail_count")
    else:
        fail_cnt = pd.Series(dtype=int)
    summary = pd.concat([success_cnt, fail_cnt], axis=1).fillna(0).astype(int)
    summary = summary.reset_index().rename(columns={"index": "task"})
    out_path = "task_sampling_summary.csv"
    summary.to_csv(out_path, index=False)
    total_fail = int(fail_df.shape[0])
    print(f"Task sampling summary saved to {out_path}, total failures: {total_fail}")


def compute_task_imbalance_metrics(train_tasks, normal_train, mode="one_vs_rest", other_abnormal_ratio=0.5):
    """Analyze class imbalance for each task in training set and return detailed results"""
    if not train_tasks:
        print("No training tasks, cannot calculate IR metrics")
        return pd.DataFrame()

    total_normal = len(normal_train)
    total_positive = sum(len(v) for v in train_tasks.values())
    records = []

    for task_key, pairs in train_tasks.items():
        family = task_key.split("__")[0]
        pos_count = len(pairs)
        other_abnormal = total_positive - pos_count
        # In one_vs_rest with oc_maml, negative sample pool consists of normal samples and other  anomaly samples grouped into
        neg_count = other_abnormal + total_normal
        min_count = min(pos_count, neg_count)
        if min_count == 0:
            ir = float("inf")
        else:
            ir = max(pos_count, neg_count) / min_count
        records.append({
            "task": task_key,
            "family": family,
            "pos_count": pos_count,
            "neg_pool_count": neg_count,
            "normal_pool_size": total_normal,
            "other_abnormal_pool": other_abnormal,
            "ir": ir,
        })

    df = pd.DataFrame(records)
    if df.empty:
        print("Cannot generate valid task imbalance statistics")
        return df

    print("\n=== Traininging Task Class Imbalance Statistics ===")
    quantiles = df["ir"].quantile([0.1, 0.5, 0.9])
    print(f"IR quantiles: P10={quantiles.loc[0.1]:.2f} | P50={quantiles.loc[0.5]:.2f} | P90={quantiles.loc[0.9]:.2f}")
    print("Most extreme tasks:")
    for _, row in df.sort_values("ir", ascending=False).head(min(5, len(df))).iterrows():
        print(
            f"  Task[{row['task']}] (family={row['family']}) Positive={row['pos_count']} | Negative pool={row['neg_pool_count']} | IR={row['ir']:.2f}"
        )
    return df


def analyze_minority_density(train_tasks, normal_train, incident_df_all, top_k=5,
                             max_pos_samples=50, max_neg_samples=200):
    """Evaluation density and neighborhood mixing of minority class in feature space"""
    if not train_tasks:
        print("No training tasks, skipped minority class density analysis")
        return []

    sorted_tasks = sorted(train_tasks.items(), key=lambda kv: len(kv[1]))
    selected = [task for task, _ in sorted_tasks[:max(1, top_k)]]
    reports = []

    print("\n=== Minority Class Density/Neighborhood Diagnosis ===")
    for task_key in selected:
        pos_pairs = train_tasks[task_key]
        pos_feats = []
        for iid, sid in pos_pairs:
            feat = extract_abnormal_features(incident_df_all, iid, sid)
            if feat is not None and feat.shape == (FEATURE_DIM,) and np.isfinite(feat).all():
                pos_feats.append(feat.astype(float))
            if len(pos_feats) >= max_pos_samples:
                break

        if len(pos_feats) < 2:
            print(f"Task[{task_key}] has < 2 positive samples, cannot evaluate density")
            continue

        neg_feats = []
        # Randomly sample some normal samples as negative reference
        normal_needed = min(len(normal_train), max(1, max_neg_samples // 2))
        if normal_train and normal_needed > 0:
            normal_indices = random.sample(range(len(normal_train)), normal_needed)
            for idx in normal_indices:
                vec = normal_train[idx]
                if isinstance(vec, np.ndarray) and vec.shape == (FEATURE_DIM,) and np.isfinite(vec).all():
                    neg_feats.append(vec.astype(float))

        # Sample anomalies from other tasks as negatives
        for other_task, pairs in train_tasks.items():
            if other_task == task_key:
                continue
            for iid, sid in pairs:
                feat = extract_abnormal_features(incident_df_all, iid, sid)
                if feat is not None and feat.shape == (FEATURE_DIM,) and np.isfinite(feat).all():
                    neg_feats.append(feat.astype(float))
                if len(neg_feats) >= max_neg_samples:
                    break
            if len(neg_feats) >= max_neg_samples:
                break

        if len(neg_feats) == 0:
            print(f"Task[{task_key}] cannot get negative features, skipped")
            continue

        pos_arr = np.stack(pos_feats)
        neg_arr = np.stack(neg_feats)

        # Calculate nearest neighbor distances within positive class
        pos_pos_diff = pos_arr[:, None, :] - pos_arr[None, :, :]
        pos_pos_dist = np.sqrt(np.maximum(np.sum(pos_pos_diff ** 2, axis=-1), 0.0))
        np.fill_diagonal(pos_pos_dist, np.inf)
        nearest_pos = pos_pos_dist.min(axis=1)

        # Calculate nearest neighbor distances from positive to negative class
        pos_neg_diff = pos_arr[:, None, :] - neg_arr[None, :, :]
        pos_neg_dist = np.sqrt(np.maximum(np.sum(pos_neg_diff ** 2, axis=-1), 0.0))
        nearest_neg = pos_neg_dist.min(axis=1)

        # Calculate negative class ratio in mixed neighborhood of positive class
        combined = np.concatenate([pos_arr, neg_arr], axis=0)
        labels = np.concatenate([np.ones(len(pos_arr)), np.zeros(len(neg_arr))])
        dist_mix = pos_arr[:, None, :] - combined[None, :, :]
        dist_mix = np.sqrt(np.maximum(np.sum(dist_mix ** 2, axis=-1), 0.0))
        for i in range(len(pos_arr)):
            dist_mix[i, i] = np.inf  # Exclude self
        k = min(5, combined.shape[0] - 1)
        neg_neighbor_ratio = []
        if k > 0:
            for i in range(len(pos_arr)):
                nearest_idx = np.argsort(dist_mix[i])[:k]
                neg_ratio = float(np.mean(labels[nearest_idx] == 0))
                neg_neighbor_ratio.append(neg_ratio)
        else:
            neg_neighbor_ratio.append(0.0)

        report = {
            "task": task_key,
            "pos_samples": len(pos_arr),
            "neg_samples": len(neg_arr),
            "median_nn_pos": float(np.median(nearest_pos)),
            "median_nn_neg": float(np.median(nearest_neg)),
            "boundary_ratio": float(np.mean(nearest_neg < nearest_pos)),
            "neg_neighbor_ratio": float(np.mean(neg_neighbor_ratio)) if neg_neighbor_ratio else 0.0,
        }
        reports.append(report)
        print(
            f"Task[{task_key}] Pos count={report['pos_samples']} | Neg samples={report['neg_samples']} | "
            f"Pos NN median={report['median_nn_pos']:.3f} | Neg NN median={report['median_nn_neg']:.3f} | "
            f"NN negative ratio={report['boundary_ratio'] * 100:.1f}% | In {k} neighborhood negassratio={report['neg_neighbor_ratio'] * 100:.1f}%"
        )

    return reports


def summarize_rare_tasks(train_tasks, normal_train, log_path="task_sampling_log.csv", top_n=5, mode="one_vs_rest"):
    """Summarize positive and negative sample sizes and sampling frequency for few-shot tasks"""
    if not train_tasks:
        print("No training tasks, cannot summarize few-shot task info")
        return pd.DataFrame()

    pos_counts = {task: len(pairs) for task, pairs in train_tasks.items()}
    sorted_tasks = sorted(pos_counts.items(), key=lambda kv: kv[1])
    selected = sorted_tasks[:max(1, top_n)]
    total_normal = len(normal_train)
    total_positive = sum(pos_counts.values())

    log_df = None
    if os.path.exists(log_path):
        try:
            log_df = pd.read_csv(log_path)
        except Exception:
            log_df = None
            print("Failed to read task sampling log, cannot summarize sampling frequency")
    else:
        print("Not found task_sampling_log.csv, only output theoretical negative sample size")

    if log_df is not None:
        success_df = log_df[log_df.get("status", "") == "success"]
        total_success = max(1, success_df.shape[0])
    else:
        success_df = None
        total_success = 1

    rows = []
    print("\n=== Few-shot Task Statistics ===")
    for task_key, pos_count in selected:
        other_abnormal = total_positive - pos_count
        neg_count = other_abnormal + total_normal
        if success_df is not None:
            freq = float((success_df.get("task") == task_key).sum() / total_success)
        else:
            freq = None
        rows.append({
            "task": task_key,
            "family": task_key.split("__")[0],
            "pos_count": pos_count,
            "neg_pool_count": neg_count,
            "sampling_freq": freq,
        })
        if freq is None:
            freq_str = "logmissing"
        else:
            freq_str = f"{freq * 100:.2f}%"
        print(
            f"Task[{task_key}] Positive={pos_count} | Negative pool={neg_count} | Theoretical IR={max(pos_count, neg_count) / max(1, min(pos_count, neg_count)):.2f} | Sampling ratio={freq_str}"
        )

    return pd.DataFrame(rows)


def _parse_int_list_field(raw) -> List[int]:
    """Parse string of log list into integer list."""
    if raw is None:
        return []
    if isinstance(raw, float) and np.isnan(raw):
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (valueError, SyntaxError):
            return []
    else:
        parsed = raw

    result: List[int] = []
    if isinstance(parsed, (list, tuple, np.ndarray)):
        for item in parsed:
            try:
                result.append(int(item))
            except (TypeError, valueError):
                continue
    elif isinstance(parsed, (int, np.integer)):
        result.append(int(parsed))
    return result


def load_proc_info_counters(log_path: str) -> Tuple[Counter, Counter, int]:
    """Summarize zero/low variance column statistics from task sampling log."""
    zero_counter = Counter(PROC_INFO_RUNTIME["zero_counts"])
    pert_counter = Counter(PROC_INFO_RUNTIME["perturb_counts"])
    total_records = PROC_INFO_RUNTIME["records"]

    if not log_path or not os.path.exists(log_path):
        return zero_counter, pert_counter, total_records

    try:
        df = pd.read_csv(log_path)
    except Exception as exc:  # pragma: no cover
        print(f"[Exp-B][Step3] Read {log_path} Failed{exc}")
        return zero_counter, pert_counter, total_records

    if "status" in df.columns:
        df = df[df["status"] == "success"]

    for _, row in df.iterrows():
        zero_list = _parse_int_list_field(
            row.get("zero_var_cols_reinserted") or row.get("zero_var_cols")
        )
        pert_list = _parse_int_list_field(
            row.get("low_var_cols_perturbed") or row.get("perturbed_cols")
        )
        zero_counter.update(zero_list)
        pert_counter.update(pert_list)
        total_records += 1

    return zero_counter, pert_counter, total_records


def _compute_family_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate statistics for each family, structure aligned with pre_resample_stats."""
    feature_cols = [col for col in df.columns if col.startswith("feat_")]
    if not feature_cols:
        return pd.DataFrame()

    pos_df = df[df["label"] == 1]
    neg_normal_df = df[df["label"] == 0]
    families = sorted(pos_df["family"].dropna().unique())

    stats_rows: List[Dict[str, object]] = []
    for fam in families:
        fam_pos = pos_df[pos_df["family"] == fam]
        other_pos = pos_df[pos_df["family"] != fam]
        fam_neg = pd.concat([neg_normal_df, other_pos], ignore_index=True)

        pos_count = int(fam_pos.shape[0])
        neg_count = int(fam_neg.shape[0])
        neg_normal_count = int(neg_normal_df.shape[0])
        neg_other_abnormal_count = int(other_pos.shape[0])
        ir = float(neg_count / pos_count) if pos_count > 0 else float("inf")

        row: Dict[str, object] = {
            "family": fam,
            "pos_count": pos_count,
            "neg_count": neg_count,
            "neg_normal_count": neg_normal_count,
            "neg_other_abnormal_count": neg_other_abnormal_count,
            "IR": ir,
        }

        if pos_count > 0:
            pos_mean = fam_pos[feature_cols].mean()
            pos_var = fam_pos[feature_cols].var(ddof=0)
        else:
            pos_mean = pd.Series([np.nan] * len(feature_cols), index=feature_cols)
            pos_var = pd.Series([np.nan] * len(feature_cols), index=feature_cols)

        if neg_count > 0:
            neg_mean = fam_neg[feature_cols].mean()
            neg_var = fam_neg[feature_cols].var(ddof=0)
        else:
            neg_mean = pd.Series([np.nan] * len(feature_cols), index=feature_cols)
            neg_var = pd.Series([np.nan] * len(feature_cols), index=feature_cols)

        for col in feature_cols:
            row[f"{col}_pos_mean"] = float(pos_mean[col]) if not np.isnan(pos_mean[col]) else np.nan
            row[f"{col}_pos_var"] = float(pos_var[col]) if not np.isnan(pos_var[col]) else np.nan
            row[f"{col}_neg_mean"] = float(neg_mean[col]) if not np.isnan(neg_mean[col]) else np.nan
            row[f"{col}_neg_var"] = float(neg_var[col]) if not np.isnan(neg_var[col]) else np.nan

        stats_rows.append(row)

    return pd.DataFrame(stats_rows)


def train_maml(
        task_dict,
        normal_pool,
        episodes=EPISODES,
        lr_inner=lr_inner,  # Inner loop learning rate
        lr_meta=lr_meta,  # Meta learning rate
        kl_scale=kl_scale,  # KL divergence scaling factor
        kl_warmup_steps=kl_warmup_steps,  # Beta Warmupof episode quantity
        inner_steps=inner_steps,
        patience=patience,  # "patience": stop training after many episodes without improvement
        task_batch=4,
        oc_align=0.1,
        first_order=False,  # [OK] Fix: use complete MAML (second-order derivatives), not FOMAML
        mode="one_vs_rest",
        device=DEVICE,
        incident_df_all=None,
        incident_source_path=None,
        normal_source_path=None,
        model_type="mlp",  # modelsType
        model_kwargs=None,  # Model-specific parameters
        resume_from=None,  # Resume from specified checkpoint
        checkpoint_dir=CHECKPOINT_DIR,  # checkpointDirectory
        add_when=session_start_when,
        run_log_dir=None,  # TimewhensRunoflogDirectory
        # ===== newly added sampling with weightconfigurationfacilitate sweep =====
    hard_neg: bool = True,  # Hard negative sample mining enabled by default (core strategy), manually disable for ablation
        other_abnormal_ratio: float = OTHER_ABNORMAL_RATIO,
        use_fixed_pos_weight: optional[bool] = None,
        fixed_pos_weight: optional[float] = None,
        # ===== newly added comparison modelssupport =====
        use_comparison_model: bool = False,  # Whether to use comparison models
        comparison_model_name: optional[str] = None,  # Comparison model name
        comparison_config: optional[dict] = None,  # Complete comparison modelconfiguration
):
    """
    integratedEd B-SMALL ofTrainingFunction
        1. Introduced kl_warmup_steps and kl_scale parameters.
        2. Use BetaScheduler for KL divergence term warmup.
        3. Replace L2 regularization loss with model's KL divergence loss (get_regularization_loss).
        4. Add sparsity report after training completes.
    """
    start_inner_whenr = when.when()  # recordEnterFunctionofwhen

    assert incident_df_all is not None, "train_maml requires incident_df_all (complete event table)"

    if model_kwargs is None:
        model_kwargs = {}

    if run_log_dir is None:
        raise valueError("train_maml necessary run_log_dir points to current run log directory")

    # createcheckpointDirectory
    os.makedirs(checkpoint_dir, exist_ok=True)

    leave_sensors = [101, 205, 309]  # Example

    '''
    # Analyze family differencesoptional
    print("\n=== Analyze Family Differences ===")
    analyze_family_diversity(task_dict, incident_df_all)
    print("=" * 40)
    elapsed_since_session = start_inner_time - add_time  # add_time is session_start_when
    print(f"currentcumulativeRunwhen: {elapsed_since_session:.2f}sec / {elapsed_since_session / 60:.2f}minutes / {elapsed_since_session / 3600:.2f}Smallwhen")
    # ======optional============
    '''

    # Split tasks & normal pools
    train_tasks, val_tasks, test_tasks = split_task_dict(
        task_dict, train_ratio=0.6, val_ratio=0.2
    )
    normal_train, normal_valtest = split_normal_pool(normal_pool)
    half = len(normal_valtest) // 2
    normal_val = normal_valtest[:half]
    normal_test = normal_valtest[half:]

    print(f"Task split: train={len(train_tasks)} | val={len(val_tasks)} | test={len(test_tasks)}")
    print(f"normal pool splitting: train={len(normal_train)} | val/test={len(normal_val) + len(normal_test)}")
    elapsed_since_session = start_inner_time - add_time  # add_time is session_start_when
    print(
        f"currentcumulativeRunwhen: {elapsed_since_session:.2f}sec / {elapsed_since_session / 60:.2f}minutes / {elapsed_since_session / 3600:.2f}Smallwhen")

    # ===== Diagnose Traininging Data Class Imbalance and Density Structure =====
    compute_task_imbalance_metrics(train_tasks, normal_train, mode=mode, other_abnormal_ratio=OTHER_ABNORMAL_RATIO)
    analyze_minority_density(train_tasks, normal_train, incident_df_all)
    summarize_rare_tasks(train_tasks, normal_train, log_path="task_sampling_log.csv", mode=mode)

    train_tasks_active = train_tasks
    normal_train_active = normal_train

    # ===== Hard Negative statusPrint =====
    print(f"[HardNeg] status={'ON' if hard_neg else 'OFF'} | Default strategy=distance |  candidate amplification factor=5 | other anomaly ratio={other_abnormal_ratio}")

    # based onconfigurationSelectmodelscreateWay
    if use_comparison_model:
        # Comparison experiments: use src.models factory function
        #  Fix: ensure project root is in Python path
        import sys
        from pathlib import Path
        project_root = Path(__file__).resolve().parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from src.models import create_model as create_comparison_model

        print(f"[Model] [OK] created comparison model: {comparison_model_name}")

        # Create simple encoder (feature extractor)
        # For comparison models, encoder should only output features, no classifier head
        class SimpleEncoder(nn.Module):
            """Simple encoder for comparison models"""
            def __init__(self, input_dim, hidden_dim=256):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                )

            def forward(self, x):
                return self.encoder(x)

        encoder = SimpleEncoder(FEATURE_DIM, hidden_dim=256)

        #  fixensure comparison_config Contains model.name
        # Ablation experiment config may lack model field, manual construction required
        if comparison_config is None:
            comparison_config = {}
        if "model" not in comparison_config or not isinstance(comparison_config.get("model"), dict):
            comparison_config["model"] = {}
        if not comparison_config["model"].get("name"):
            comparison_config["model"]["name"] = comparison_model_name
            print(f"[Model]  Automaticset model.name = {comparison_model_name}")

        # Create comparison models
        model_instance = create_comparison_model(comparison_config, encoder)

        print(f"[Model] comparison model created successfully, type: {type(model_instance).__name__}")
    else:
        # Ablation experiments: use original SABLE model factory
        model_instance = get_model(model_type=model_type, input_dim=FEATURE_DIM, model_kwargs=model_kwargs)
        if not isinstance(model_instance, BaseModel):
            raise TypeError(
                f"Model factory returned non-BaseModel instance ({type(model_instance)}), please check {model_type}  registrationconfiguration"
            )

    model = model_instance.to(device)
    # Calculate KL warmup steps: if not explicitly specified (<=0), automatically use 20% of total episodesticCalculate
    effective_kl_warmup = int(kl_warmup_steps) if (kl_warmup_steps and kl_warmup_steps > 0) else max(1, int(episodes * 0.2))
    kl_scheduler = BetaScheduler(warmup_steps=effective_kl_warmup)
    print(f"[KL Warmup] UselinearWarmup: warmup_steps(effective)={effective_kl_warmup}")
    # ===== Comparison models Special Processing: All comparison models use standard CE loss =====
    # MAML/ProtoNet/TSA-MAML papers all require standard cross-entropy loss
    # Focal loss dynamic weighting conflicts with meta-learning, causing model degradation
    # NOTE: Config file may write TSA-MAML as TSA_MAML or TSA-MAML, need to handle bothnSupport
    if use_comparison_model and comparison_model_name in ["MAML", "ProtoNet", "TSA-MAML", "TSA_MAML"]:
        import torch.nn.functional as F
        class CELoss(nn.Module):
            """Standard cross-entropy loss (for comparison models only)"""
            def __init__(self):
                super().__init__()
            def forward(self, logits, targets):
                return F.cross_entropy(logits, targets, reduction='mean')
        base_loss_fn = CELoss()
        print(f"[{comparison_model_name}] [OK] using standard cross-entropy loss (matches paper requirement)")
    else:
        # SABLE and other models use Focal Loss for class imbalance
        base_loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
        print("[SABLE] using Focal Loss for class imbalance")
    # ====================================================

    # ===== Standard MAML OptimizerUsefixedofMeta learning rate =====
    meta_opt = torch.optim.Adam(model.parameters(), lr=lr_meta)

    print(f"Usemodels: {model_type.upper()}")
    elapsed_since_session = start_inner_time - add_time  # add_time is session_start_when
    print(
        f"currentcumulativeRunwhen: {elapsed_since_session:.2f}sec / {elapsed_since_session / 60:.2f}minutes / {elapsed_since_session / 3600:.2f}Smallwhen")

    # InitializeTrainingstatus
    start_episode = 1
    best_val = -1.0  # Compatible with keep:history based on accuracy optimal value (no longer used for optimization)
    best_score = -1.0  # newly addedbased on recall floor constraint F_beta@t* as early stopping/save criterion
    best_threshold = None  # newly addedvalidation phase selected optimal threshold value (frozen for test set)
    best_rmin = None       # newly addedcorresponding R_min at optimal model time
    best_state = None

    # ====== newly addedtimer initialization ======
    checkpoint = None
    accumulated_training_when = 0.0
    # ==============================

    # ===== early stopping mechanism initialization =====
    epochs_no_improve = 0  # Initialize counter:record performance non-improvement epoch count
    # ===============================

    # ===== Task samplinglog =====
    task_sampling_logs = []
    # ===============================

    # ===== metricsloglist =====
    train_log = []  # record training set metrics
    val_log = []  # record validation set metrics
    val_family_log = []  # recordValidation setbased on family ofmetrics
    final_val_results = None  # EventuallyValidation setEvaluationResults
    final_test_results = None  # EventuallyTest setEvaluationResults
    # ===== unified timestamps for log and metrics alignment =====
    abs_run_log_dir = os.path.abspath(run_log_dir)
    whenstamp_str = os.path.basename(abs_run_log_dir.rstrip(os.sep))
    # ===== TensorBoard logger =====
    writer = SummaryWriter(log_dir=abs_run_log_dir)
    # ===== metricsSaveDirectory =====
    metrics_dir = os.path.join("metrics", f"{whenstamp_str}-expB")
    os.makedirs(metrics_dir, exist_ok=True)
    # ===== ClassweightlogDirectory =====
    class_weight_records = []
    class_weight_log_path = None
    if USE_CLASS_WEIGHT:
        class_weight_dir = metrics_dir
        class_weight_log_path = os.path.join(class_weight_dir, "class_weight_log.csv")
        print(f"[Exp-B][Step8] EnabledClassweightlog see {class_weight_log_path}")
    else:
        print("[Exp-B][Step8] kept closed, not generating class_weight_log.csv")
    # ===============================

    # check if there ischeckpointnecessaryResume
    if USE_CHECKPOINT:
        if resume_from is not None:
            checkpoint_path = os.path.join(checkpoint_dir, resume_from)
            if os.path.exists(checkpoint_path):
                print(f"fromcheckpointResume: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device)
            else:
                print(f"WARNING: Checkpoint file {checkpoint_path} does not exist, starting training from beginning")
        else:
            checkpoint_files = [
                f for f in os.listdir(checkpoint_dir)
                if f.startswith('checkpoint_') and f.endswith('.pth')
            ]
            if checkpoint_files:
                latest_checkpoint = max(
                    checkpoint_files,
                    key=lambda x: int(x.split('_')[1].split('.')[0])
                )
                checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                print(f"AutomaticResumerecentofcheckpoint: {latest_checkpoint}")
                checkpoint = torch.load(checkpoint_path, map_location=device)
            else:
                print("notdiscoveravailableofCheckpoint filestart from beginningTraining")

        if checkpoint is not None:
            model.load_state_dict(checkpoint['model_state_dict'])
            meta_opt.load_state_dict(checkpoint['optimizer_state_dict'])
            start_episode = checkpoint['episode'] + 1
            best_val = checkpoint['best_val']
            if 'best_model_state' in checkpoint:
                best_state = checkpoint['best_model_state']
            # ResumeRecall priority related threshold value/score (if exists)
            best_threshold = checkpoint.get('best_threshold', None)
            best_rmin = checkpoint.get('best_rmin', None)
            best_score = checkpoint.get('best_fbeta', best_score if 'best_fbeta' in locals() else -1.0)
            prev_thr = checkpoint.get('selected_threshold', None)
            # stepwise threshold increase status (ignore if not used)
            rmin_current = checkpoint.get('rmin_current', R_START)
            # history optional
            # recall_history = checkpoint.get('recall_history', [])

            print(f"fromNo. {start_episode} roundContinueTrainingcurrentoptimalValidation setPerformanceance(Acc): {best_val:.3f}")
            if best_threshold is not None:
                print(f"Resumethreshold valueINFObest_fbeta={best_score if best_score is not None else -1:.3f}, "
                      f"best_rmin={best_rmin if best_rmin is not None else float('nan'):.3f}, "
                      f"best_threshold={best_threshold:.3f}")
    else:
        print("DisabledcheckpointResumestart from beginningTraining")

    # decouple log and evaluation frequency; increase validation episodes to stabilize threshold value estimation
    log_every = LOG_EVERY
    eval_every = EVAL_EVERY
    eval_episodes = 200 if not SANITY else 10
    # threshold value selection status (for EMA smoothing, etc.)
    prev_thr = None
    rmin_current = R_START  # stepwise threshold increase current value (only as placeholder in linear mode)
    recall_history = []     # save recent several times Recall@t*
    total_eps = episodes if not SANITY else min(episodes, 10)

    # TrainingStartingwhenrecordwhenSupportfromcheckpointResume
    if 'checkpoint' in locals() and checkpoint is not None and 'accumulated_training_when' in checkpoint:
        # fromcheckpointResumecumulativeTrainingwhen
        accumulated_training_when = checkpoint['accumulated_training_when']
        print(
            f"ResumecumulativeTrainingwhen: {accumulated_training_when:.2f}sec / {accumulated_training_when / 60:.2f}minutes / {accumulated_training_when / 3600:.2f}Smallwhen / {accumulated_training_when / 86400:.2f}Day")
    else:
        # completely start training
        accumulated_training_when = 0.0
        print("Starting training from beginning, training timer starts")

    #record current episode's training session start time
    session_start_when = when.when()

    # Determine fixed weight strategy for this episode (use global config if not specified)
    use_fixed_weight_flag = USE_FIXED_POS_WEIGHT if use_fixed_pos_weight is None else bool(use_fixed_pos_weight)
    fixed_weight_value = float(FIXED_POS_WEIGHT if fixed_pos_weight is None else fixed_pos_weight)



    for ep in range(start_episode, total_eps + 1):
        model.train()

        # ===== B-SMALL: Get KL divergence weight for current episode =====
        kl_beta = kl_scheduler.get_beta(current_step=ep)
        # ==================================================

        # Original meta_opt.zero_grad() moved after loss calculation, execute after gradientte
        meta_losses, meta_accs = [], []

        # Collect query predictions and labels from all tasks for recall/F1 statistics
        q_logits_list = []
        qy_list = []

        for _ in range(task_batch):
            # Retry: try to get a valid task; skip this task if unavailable
            got = False
            last_fail_task = None
            for _retry in range(8):
                try:
                    sample_result = sample_task(
                        task_dict=train_tasks_active,
                        current_split_tasks=train_tasks_active,
                        normal_pool=normal_train_active,
                        mode=mode,
                        incident_df_all=incident_df_all,
                        debug=False,
                        other_abnormal_ratio=other_abnormal_ratio,  # Ratio of other anomalies in negative samples
                        hard_neg=hard_neg,
                    )
                    sx, sy, qx, qy, info = _unwrap_sample_task(sample_result)
                    got = True
                    break
                except RunwhenError as e:
                    last_fail_task = e.args[0] if len(e.args) > 0 else None
                    continue
            if not got:
                # Record task that failed sampling on last attempt
                task_sampling_logs.append({
                    "episode": ep,
                    "status": "failed",
                    "task": last_fail_task,
                })
                continue  # Skip this small task, continue to other tasks

            # Record successfully sampled task info
            info.update({"episode": ep, "status": "success"})
            task_sampling_logs.append(info)

            sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)

            # ===== Class Weight Calculation (only effective when enabled in config) =====
            class_weights_tensor = None
            if USE_CLASS_WEIGHT:
                if use_fixed_weight_flag:
                    pos_query_count = int((qy == 1).sum().item())
                    neg_query_count = int((qy == 0).sum().item())
                    raw_ratio = float('nan')
                    capped_flag = 0
                    pos_weight_value = float(fixed_weight_value)
                    class_weights_tensor = torch.tensor(
                        [1.0, pos_weight_value], dtype=torch.float32, device=device
                    )
                    class_weight_records.append({
                        "episode": ep,
                        "task": info.get("task"),
                        "family": info.get("family"),
                        "pos_query": pos_query_count,
                        "neg_query": neg_query_count,
                        "raw_ratio": raw_ratio,
                        "pos_weight": pos_weight_value,
                        "capped": capped_flag,
                        "fixed": 1,
                    })
                else:
                    pos_query_count = int((qy == 1).sum().item())
                    neg_query_count = int((qy == 0).sum().item())
                    if pos_query_count <= 0:
                        raw_ratio = float('inf')
                        pos_weight_value = 10.0
                        capped_flag = 1
                    else:
                        raw_ratio = neg_query_count / max(pos_query_count, 1)
                        pos_weight_value = min(10.0, raw_ratio)
                        capped_flag = int(pos_weight_value < raw_ratio)
                    class_weights_tensor = torch.tensor(
                        [1.0, float(pos_weight_value)], dtype=torch.float32, device=device
                    )
                    class_weight_records.append({
                        "episode": ep,
                        "task": info.get("task"),
                        "family": info.get("family"),
                        "pos_query": pos_query_count,
                        "neg_query": neg_query_count,
                        "raw_ratio": raw_ratio,
                        "pos_weight": float(pos_weight_value),
                        "capped": capped_flag,
                        "fixed": 0,
                    })
            # ===== Comparison models Special Processing: Select loss function based on model type =====
            # MAML/ProtoNet/TSA-MAMLUseStandardCElossOutput[N, 2]necessaryLongTypetarget
            # JointTraininging/LogRegUseBCEwithLogitsLossOutput[N]necessaryFloatTypetarget
            # SABLE and other models: use Focal Loss for class imbalance
            if use_comparison_model and comparison_model_name in ["MAML", "ProtoNet", "TSA-MAML", "TSA_MAML"]:
                # Meta-learning comparison models: use standard CE loss
                import torch.nn.functional as F
                loss_fn_task = lambda logits, targets: F.cross_entropy(
                    logits, targets.long(), reduction='mean'
                )
                print(f"[{comparison_model_name}] [OK] loss_fn_task using standard cross-entropy loss")
            elif use_comparison_model and comparison_model_name in ["JointTraininging", "LogReg"]:
                # Joint training models: use BCEWithLogitsLoss (single logit output, requires float target)
                import torch.nn.functional as F
                loss_fn_task = lambda logits, targets: F.binary_cross_entropy_with_logits(
                    logits.squeeze(-1), targets.float(), reduction='mean'
                )
                print(f"[{comparison_model_name}] [OK] loss_fn_taskUseBCEwithLogitsLoss")
            else:
                # SABLE and other models: use Focal Loss for class imbalance
                loss_fn_task = FocalLoss(
                    alpha=0.25, gamma=2.0, reduction='mean',
                    class_weights=class_weights_tensor
                ) if class_weights_tensor is not None else base_loss_fn
            # ====================================================

            # ====== Comparison models Special Processing: ProtoNet/TSA-MAML use episodic training ======
            # NOTE: Config file may write TSA-MAML as TSA_MAML or TSA-MAML, need to handle bothnSupport
            if use_comparison_model and comparison_model_name in ["ProtoNet", "TSA-MAML", "TSA_MAML"]:
                # ProtoNetAndTSA-MAMLUseEachofepisodicTrainingMethods
                # ConstructdictionaryFormatofsupportAndquery
               support = {'x': sx, 'y': sy}
                query = {'x': qx, 'y': qy}

                # adjustusemodelsoftrain_episodeMethods
                # NOTEtrain_episode()Internalcompleteddbackward()Andoptimizer.step()
                episode_metrics = model.train_episode(support, query)

                # Collect metrics
                task_loss_value = episode_metrics['loss']  # Python float
                acc_value = episode_metrics['accuracy']     # Python float

                # Construct virtual tensors for compatibility with existing code (not for backpropagation)
                # train_episode already updated parameters internally, here only for logging
                task_loss = torch.tensor(task_loss_value, device=device, requires_grad=False)

                # Collect for logging
                meta_losses.append(task_loss)
                meta_accs.append(torch.tensor(acc_value, device=device))

                # Collect for overall metrics calculation (requires forward pass to get logits)
                with torch.no_grad():
                    q_logits = model.forward(qx)  # SimpleforwardGetlogitsFormetricsstatistics
                    q_logits_list.append(q_logits.cpu())
                    qy_list.append(qy.cpu())
            else:
                # ====== OC gradient for alignment term (at initial weights) ======
                align_term = 0.0
                if mode == "oc_maml" and oc_align > 0:
                    init_fast = model.named_params_dict()
                    # ===== Comparison models Special Processing: MAML/ProtoNet/TSA-MAML use standard CE loss =====
                    # Comparison model papers require standard cross-entropy loss, do not use Focal Loss
                    # NOTE: Config file may write TSA-MAML as TSA_MAML or TSA-MAML, need to handle bothnSupport
                    if use_comparison_model and comparison_model_name in ["MAML", "ProtoNet", "TSA-MAML", "TSA_MAML"]:
                        # Comparison models: use standard CE loss (ignore class_weights_tensor)
                        import torch.nn.functional as F
                        loss_fn_align = lambda logits, targets: F.cross_entropy(
                            logits, targets.long(), reduction='mean'
                        )
                        print(f"[{comparison_model_name}] [OK] loss_fn_align using standard cross-entropy loss")
                    else:
                        # SABLE and other models: use Focal Loss for class imbalance
                        loss_fn_align = FocalLoss(
                            alpha=0.25, gamma=2.0, reduction='mean',
                            class_weights=class_weights_tensor
                        ) if class_weights_tensor is not None else base_loss_fn
                    # ====================================================

                    # Support set (OC) gradient (no second-order graph, only for alignment regularization)
                    logits_s0 = model.forward_with_weights(sx, init_fast)
                    loss_s0 = loss_fn_align(logits_s0, sy)
                    g_oc = torch.autograd.grad(loss_s0, list(init_fast.values()), create_graph=False)

                    # Query set (CB) gradient (also at initial point)
                    logits_q0 = model.forward_with_weights(qx, init_fast)
                    loss_q0 = loss_fn_align(logits_q0, qy)
                    g_cb = torch.autograd.grad(loss_q0, list(init_fast.values()), create_graph=False)

                    cos = _cosine_sim(g_oc, g_cb)
                    # negative sign: our"maximization" cost is equivalent to subtracting this item at meta loss
                    align_term = - oc_align * cos

                # ====== Standard MAML inloopUpdate ======
                fast = model.named_params_dict()
                for _step in range(inner_steps):
                    logits_s = model.forward_with_weights(sx, fast)
                    task_loss_s = loss_fn_task(logits_s, sy)

                    # ===== B-SMALL: inlooploss = Taskloss + KL divergenceloss =====
                    kl_loss_s = model.get_regularization_loss() if hasattr(model, 'get_regularization_loss') else 0.0
                    loss_s = task_loss_s + kl_beta * kl_scale * kl_loss_s
                    # =====================================================

                    # Getinloopgradient
                    grads = torch.autograd.grad(
                        loss_s, list(fast.values()),
                        create_graph=not first_order,
                        retain_graph=True,
                        allow_unused=True
                    )

                    # UsefixedLearning rateUpdateFastweight
                    new_fast = OrderedDict()
                    grad_idx = 0
                    for n, w in fast.items():
                        if w.requires_grad:
                            g = grads[grad_idx]
                            grad_idx += 1
                            if g is not None:
                                new_fast[n] = w - lr_inner * g
                            else:
                                new_fast[n] = w
                        else:
                            new_fast[n] = w
                    fast = new_fast

                # ====== Standard MAML metalossCalculateoutloop ======
                q_logits = model.forward_with_weights(qx, fast)
                task_loss_q = loss_fn_task(q_logits, qy)

                # ===== B-SMALL: metaloss = Taskloss + KL divergenceloss =====
                kl_loss_q = model.get_regularization_loss() if hasattr(model, 'get_regularization_loss') else 0.0
                task_loss = task_loss_q + kl_beta * kl_scale * kl_loss_q
                # =================================================

                task_loss = task_loss + (align_term if isinstance(align_term, torch.Tensor) else 0.0)
                meta_losses.append(task_loss)
                #  fixbased onmodelsOutputFormatCalculateAccuracyJointTraininging/LogRegCompatible with
                if use_comparison_model and comparison_model_name in ["JointTraininging", "LogReg"]:
                    # Single logit Output[N]: Use sigmoid > 0.5 as threshold for predicting Positive class
                    preds = (torch.sigmoid(q_logits.squeeze(-1)) > 0.5).long()
                    meta_accs.append((preds == qy).float().mean())
                else:
                    # Dual-class logits Output[N, 2]Useargmax
                    meta_accs.append((q_logits.argmax(1) == qy).float().mean())

                # collect for overall metrics
                q_logits_list.append(q_logits.detach().cpu())
                qy_list.append(qy.detach().cpu())

        if not meta_losses:
            if SANITY: print("[warn] episode skipped (0 tasks collected)")
            continue

        # ====== metaUpdate ======
        # ProtoNet and TSA-MAML's train_episode() already internally completed parameter update
        # OthermodelsSABLE/MAMLnecessaryStandardMAMLofmetaUpdate
        if not (use_comparison_model and comparison_model_name in ["ProtoNet", "TSA-MAML", "TSA_MAML"]):
            meta_opt.zero_grad()
            meta_loss = torch.stack(meta_losses).mean()
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            meta_opt.step()
        else:
            # ProtoNet/TSA-MAML: meta_lossForlogrecord
            meta_loss = torch.stack(meta_losses).mean()
        # ========================================

        acc = torch.stack(meta_accs).mean().item() if meta_accs else float('nan')

        # Uniformly statistics query set metrics of all tasks
        with torch.no_grad():
            if q_logits_list:
                all_q_logits = torch.cat(q_logits_list, dim=0)
                all_qy = torch.cat(qy_list, dim=0)

                #  fixbased onmodelsOutputFormatCalculateprobabilityAndPredictionJointTraininging/LogRegCompatible with
                if all_q_logits.ndim == 2 and all_q_logits.shape[1] == 2:
                    # Dual-class logits Output[N, 2]
                    probs = torch.softmax(all_q_logits, dim=1)[:, 1].numpy()
                    preds = all_q_logits.argmax(1).numpy()
                else:
                    # Single logit Output[N]JointTraininging/LogReg
                    probs = torch.sigmoid(all_q_logits.squeeze(-1)).numpy()
                    preds = (probs > 0.5).astype(int)
                labels = all_qy.numpy()

                try:
                    recall = recall_score(labels, preds)
                except Exception:
                    recall = 0.0
                try:
                    f1 = f1_score(labels, preds)
                except Exception:
                    f1 = 0.0
            else:
                recall, f1 = 0.0, 0.0
        # ===== Modify: Print alpha and beta values for monitoring =====
        # ===== PrintTrainingmetrics =====
        if ep % log_every == 0 or ep == 1:
            print(
                f"Episode {ep}: Meta Loss={meta_loss.item():.4f}, Acc={acc:.2f}, Recall={recall:.3f}, F1={f1:.3f}, "
                f"KL_beta={kl_beta:.3f}"
            )
            # ================================================
            # progress JSON (Training phase no validation set aggregate metricsuse current training metrics write out
            try:
                if PROGRESS_JSONL_PATH and (ep == 1 or ep % PROGRESS_EVAL_INTERVAL == 0):
                    report_progress(
                        episode=ep,
                        episodes_total=total_eps,
                        elapsed_sec=when.when() - session_start_when,
                        metrics={"f1": f1, "recall": recall, "precision": None, "threshold": None},
                        config_path=CLI_CONFIG_PATH,
                        run_dir=run_log_dir,
                    )
            except Exception:
                pass
        # ========================
        # ===== SaveTrainingmetrics =====
        train_log.append({
            "episode": ep,
            "meta_loss": meta_loss.item(),
            "acc": acc,
            "recall": recall,
            "f1": f1,
        })
        # ===== TensorBoard recordTrainingmetrics =====
        writer.add_scalar("train/meta_loss", meta_loss.item(), ep)
        writer.add_scalar("train/acc", acc, ep)
        writer.add_scalar("train/recall", recall, ep)
        writer.add_scalar("train/f1", f1, ep)
        # ===============================================

        # Every few rounds at validation set on evaluation
        if ep % eval_every == 0 and len(val_tasks) > 0:
            val_results = evaluate_on_split(
                model, val_tasks, normal_val, episodes=eval_episodes,
                device=device, lr_inner=lr_inner, first_order=first_order,
                mode=mode, incident_df_all=incident_df_all,
                show_prob_hist=True,
                search_best_threshold=False,
                hard_neg=hard_neg,
                other_abnormal_ratio=other_abnormal_ratio,
            )
            val_acc = val_results["acc"]  # keepPrintReference

            # CalculatecumulativeRunwhen
            current_when = when.when()
            elapsed_when = current_when - session_start_when
            total_elapsed = accumulated_training_when + elapsed_when
            print(f"  Val split Acc={val_acc:.3f}, Recall={val_results['recall']:.3f}, "
                  f"ROC-AUC={val_results['roc_auc']:.3f}, PR-AUC={val_results['pr_auc']:.3f}, "
                  f"F1={val_results['f1']:.3f}")
            for fam, fm in val_results.get("family_metrics", {}).items():
                print(f"    Family {fam}: Acc={fm['acc']:.3f}, Recall={fm['recall']:.3f}, F1={fm['f1']:.3f}")
                val_family_log.append({
                    "episode": ep,
                    "family": fam,
                    "acc": fm["acc"],
                    "recall": fm["recall"],
                    "f1": fm["f1"],
                })
            if total_elapsed == elapsed_when:
                print(
                    f"  cumulativeTrainingwhen: {total_elapsed:.2f}sec / {total_elapsed / 60:.2f}minutes / {total_elapsed / 3600:.2f}Smallwhen / {total_elapsed / 86400:.2f}Day")
            else:
                print(
                    f"  current episode'sTrainingwhen: {elapsed_when:.2f}sec / {elapsed_when / 60:.2f}minutes) / {elapsed_when / 3600:.2f}Smallwhen / {elapsed_when / 86400:.2f}Day")
                print(
                    f"  cumulativeTrainingwhen: {total_elapsed:.2f}sec / {total_elapsed / 60:.2f}minutes) / {total_elapsed / 3600:.2f}Smallwhen / {total_elapsed / 86400:.2f}Day")

            # ===== SaveValidationmetrics =====
            val_log.append({
                "episode": ep,
                "meta_loss": val_results.get("meta_loss", 0.0),
                "acc": val_results["acc"],
                "recall": val_results["recall"],
                "f1": val_results["f1"],
            })

            # ===== TensorBoard recordValidationmetrics =====
            writer.add_scalar("val/meta_loss", val_results.get("meta_loss", 0.0), ep)
            writer.add_scalar("val/acc", val_results["acc"], ep)
            writer.add_scalar("val/recall", val_results["recall"], ep)
            writer.add_scalar("val/f1", val_results["f1"], ep)
            # ======================================

            # ==============================================

            # ===== Recall prioritythreshold valueSelectwithrecord =====
            # take validation aggregate probabilities and labelsevaluate_on_split already returned
            y_prob_all = np.array(val_results.get("probabilities", []), dtype=float)
            y_true_all = np.array(val_results.get("labels", []), dtype=int)

            # Calculate current R_minDefault linear threshold increase, can switch for staircase or disabled
            if RMIN_SCHEDULE == "disabled":
                # Recall priority strategy disabled, set r_min=0no recall constraint
                r_min = 0.0
            elif RMIN_SCHEDULE == "linear":
                r_min = get_rmin_linear(ep, total_eps, R_START, R_TARGET, R_WARMUP_FRAC)
            else:
                r_min = float(rmin_current)

            # based on USE_THRESHOLD_SEARCH switchselect thresholdsholdstrategy
            if not USE_THRESHOLD_SEARCH:
                # disabledThreshold searchUsefixed threshold
                sel = {
                    "threshold": FIXED_THRESHOLD,
                    "recall": 0.0,  # placeholder, need actual calculation
                    "precision": 0.0,  # placeholder
                    "fbeta": 0.0,  # placeholder
                    "pos_rate": 0.0,  # placeholder
                    "feasible": False,  # fix: empty input when keep Falsewith threshold search function consistenttent
                }
                # Calculatefixed thresholdunderofmetrics
                y_pred_fixed = (y_prob_all >= FIXED_THRESHOLD).astype(int)
                if len(y_true_all) > 0:
                    sel["feasible"] = True  # Set feasible only when data available
                    sel["recall"] = float(np.sum((y_pred_fixed == 1) & (y_true_all == 1)) / max(1, np.sum(y_true_all == 1)))
                    sel["precision"] = float(np.sum((y_pred_fixed == 1) & (y_true_all == 1)) / max(1, np.sum(y_pred_fixed == 1)))
                    sel["pos_rate"] = float(np.mean(y_pred_fixed))
                    if sel["recall"] + sel["precision"] > 0:
                        sel["fbeta"] = (1 + FBETA_BETA**2) * sel["precision"] * sel["recall"] / (FBETA_BETA**2 * sel["precision"] + sel["recall"])
            else:
                # enabledThreshold searchuse adaptive thresholdSelect
                sel = select_threshold_with_recall_floor(
                    y_true_all, y_prob_all,
                    r_min=r_min, beta=FBETA_BETA,
                    prev_thr=prev_thr, ema=THRESH_EMA,
                    pos_rate_cap=POS_RATE_CAP,
                )
            prev_thr = sel["threshold"]
            recall_history.append(sel["recall"])

            # Stepwise threshold increase (backup)
            if RMIN_SCHEDULE == "ladder":
                new_r = get_rmin_ladder(recall_history, rmin_current, R_TARGET,
                                         LADDER_WINDOW, LADDER_MARGIN, LADDER_STEP)
                if new_r > rmin_current:
                    print(f"  [Stepwise threshold increase] R_min: {rmin_current:.3f} -> {new_r:.3f}")
                    rmin_current = new_r

            # PrintRecall prioritymetrics
            print(
                f"  [Recall priority] R_min={r_min:.3f}, t*={sel['threshold']:.3f}, "
                f"Recall@t*={sel['recall']:.3f}, F{int(FBETA_BETA)}@t*={sel['fbeta']:.3f}, "
                f"Precision@t*={sel['precision']:.3f}, PosRate@t*={sel['pos_rate']:.3f}, "
                f"feasible={sel['feasible']}"
            )

            # recordto TensorBoardIf available
            try:
                writer.add_scalar("val/threshold", sel["threshold"], ep)
                writer.add_scalar("val/r_min", r_min, ep)
                writer.add_scalar("val/recall_tstar", sel["recall"], ep)
                writer.add_scalar("val/precision_tstar", sel["precision"], ep)
                writer.add_scalar("val/fbeta_tstar", sel["fbeta"], ep)
                writer.add_scalar("val/pos_rate_tstar", sel["pos_rate"], ep)
            except Exception:
                pass

            # ===== Progress JSON (Validation Phase: includes current threshold search metrics) =====
            try:
                if PROGRESS_JSONL_PATH and (ep % PROGRESS_EVAL_INTERVAL == 0):
                    report_progress(
                        episode=ep,
                        episodes_total=total_eps,
                        elapsed_sec=when.when() - session_start_when,
                        metrics={
                            "f1": val_results.get("f1"),
                            "recall": val_results.get("recall"),
                            "precision": sel.get("precision") if 'sel' in locals() else None,
                            "threshold": sel.get("threshold") if 'sel' in locals() else None,
                            "auc": val_results.get("roc_auc"),
                        },
                        config_path=CLI_CONFIG_PATH,
                        run_dir=run_log_dir,
                    )
            except Exception:
                pass
            # ===== Early stopping logic: use F_beta@t* under feasible conditions as criterion =====
            if sel["feasible"] and sel["fbeta"] > best_score:
                best_score = sel["fbeta"]
                best_threshold = sel["threshold"]
                best_rmin = r_min
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(
                    f"  [Early stopping criterion] Improved: F{int(FBETA_BETA)}@t*={best_score:.3f} (R_min={best_rmin:.3f}, t*={best_threshold:.3f}), alreadySaveoptimalmodels"
                )
                epochs_no_improve = 0
            else:
                epochs_no_improve += eval_every
                reason = "Infeasible (Recall<R_min)" if not sel["feasible"] else "No improvement"
                if USE_EARLY_STOPPING:
                    print(f"  [Early stopping criterion] {reason}. Epochs until early stopping: {patience - epochs_no_improve} round.")

            # Only check early stopping conditions when early stopping enabled (supports pilot ablationneTrainingepoch
            if USE_EARLY_STOPPING and epochs_no_improve >= patience:
                print(f"\n  Validation set did not improve for {epochs_no_improve} epochs (based on F_beta@t* criterion), No.{ep} round triggerearly stopping")
                break
            # ==========================

        # AutomaticSavecheckpoint
        if ep % SAVE_EVERY == 0:
            # atSavecheckpointwhenAddTrainingwhenINFO
            current_when = when.when()
            current_session_when = current_when - session_start_when
            total_accumulated_when = accumulated_training_when + current_session_when
            checkpoint = {
                'episode': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': meta_opt.state_dict(),
                'best_val': best_val,
                'best_model_state': best_state,
                # Recall priorityRelatedofstatus
                'selected_threshold': prev_thr,
                'best_threshold': best_threshold,
                'best_rmin': best_rmin,
                'best_fbeta': best_score,
                'model_type': model_type,
                'model_kwargs': model_kwargs,
                'lr_inner': lr_inner,
                'lr_meta': lr_meta,
                'kl_warmup_steps': kl_warmup_steps,
                'kl_scale': kl_scale,
                'accumulated_training_time': total_accumulated_time,  # Cumulative training time
                'session_start_time': session_start_time,  # Current session start time
                'inner_steps': inner_steps,
                'mode': mode
            }
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{ep}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f" Checkpoint saved: {checkpoint_path}")

            # Clean old checkpoints, keep only recent 3
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if
                                f.startswith('checkpoint_') and f.endswith('.pth')]
            if len(checkpoint_files) > 3:
                checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
                for old_file in checkpoint_files[:-3]:
                    old_path = os.path.join(checkpoint_dir, old_file)
                    try:
                        os.remove(old_path)
                        print(f"  Cleaned old checkpoint: {old_file}")
                    except OSError:
                        pass

    # Traininging completeafterLoadoptimalmodels
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded optimal validation performance model (Val Acc={best_val:.3f})")

    # SaveEventuallymodels
    final_checkpoint = {
        'episode': ep,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': meta_opt.state_dict(),
        'best_val': best_val,
        'best_model_state': best_state,
        'model_type': model_type,
        'model_kwargs': model_kwargs,
        'lr_inner': lr_inner,
        'lr_meta': lr_meta,
        'kl_warmup_steps': kl_warmup_steps,
        'kl_scale': kl_scale,
        'inner_steps': inner_steps,
        'mode': mode,
        'training_complete': True
    }
    final_path = os.path.join(checkpoint_dir, 'final_model.pth')
    torch.save(final_checkpoint, final_path)
    print(f" Final model saved: {final_path}")

    # ===== SaveTask samplinglog =====
    if task_sampling_logs:
        pd.DataFrame(task_sampling_logs).to_csv('task_sampling_log.csv', index=False)
        print("Task sampling log saved to task_sampling_log.csv")
        analyze_task_sampling_log('task_sampling_log.csv')

    # ===== Save and plot training and validation metrics =====
    save_metrics(train_log, val_log, out_dir=metrics_dir)
    plot_metrics(train_log, val_log, out_dir=metrics_dir)
    save_family_metrics(val_family_log, os.path.join(metrics_dir, "val_family_metrics.csv"))
    if class_weight_log_path and class_weight_records:
        pd.DataFrame(class_weight_records).to_csv(class_weight_log_path, index=False)
        print(f"Saved class weight log to {class_weight_log_path}")
    print(f"Traininging and validation metrics saved to {metrics_dir} directory")

    # ===== Final validation set evaluation for generating comparison plots =====
    if len(val_tasks) > 0:
        final_val_results = evaluate_on_split(
            model,
            val_tasks,
            normal_val,
            episodes=eval_episodes,
            device=device,
            lr_inner=lr_inner,
            first_order=first_order,
            mode=mode,
            incident_df_all=incident_df_all,
            show_prob_hist=True,
        )
        save_family_summary(
            final_val_results.get("family_metrics", {}),
            os.path.join(metrics_dir, "val_family_metrics_B.csv"),
        )
        plot_eval_artifacts(
            final_val_results.get("probabilities", []),
            final_val_results.get("labels", []),
            final_val_results.get("predictions", []),
            metrics_dir,
            "val",
        )

    # EventuallyatTest setonEvaluationOnce
    if len(test_tasks) > 0:
        # ===== Determine method name (for saving data files) =====
        current_method_name = COMPARISON_MODEL_NAME if USE_COMPARISON_MODEL else "SABLE"

        test_results = evaluate_on_split(
            model, test_tasks, normal_test, episodes=eval_episodes,
            device=device, lr_inner=lr_inner, first_order=first_order,
            mode=mode, incident_df_all=incident_df_all,  #  Passed in
            show_prob_hist=True,
            hard_neg=hard_neg,
            other_abnormal_ratio=other_abnormal_ratio,
            fixed_threshold=(best_threshold if best_threshold is not None else None),
            # ===== Save PR Curve data (2026-03-03 ) =====
            save_scores=True,
            method_name=current_method_name,
            output_dir="outputs/fig2",
            seed=SEED,  # 2026-03-03 pass seed for file naming
        )
        # Summary after training completes (must be placed after training loop)
        training_end_when = when.when()
        current_session_when = training_end_when - session_start_when
        total_training_when = accumulated_training_when + current_session_when
        print(f"\n========== EventuallyTest setResults ==========")
        print(f"  Acc={test_results['acc']:.3f}")
        print(f"  Recall (miss rate)={test_results['recall']:.3f}")
        print(f"  ROC-AUC (classification capability under class imbalance)={test_results['roc_auc']:.3f}")
        print(f"  PR-AUC (classification capability under class imbalance)={test_results['pr_auc']:.3f}")
        print(f"  F1 (average of missed and false reports)={test_results['f1']:.3f}")
        for fam, fm in test_results.get("family_metrics", {}).items():
            print(f"    Family {fam}: Acc={fm['acc']:.3f}, Recall={fm['recall']:.3f}, F1={fm['f1']:.3f}")
        print(f"=====================================")
        # ===================================

        if total_training_when == current_session_when:
            print(
                f"  current episode's total training time: {current_session_when:.2f}sec / {current_session_when / 60:.2f}minutes) / {current_session_when / 3600:.2f}Smallwhen / {current_session_when / 86400:.2f}Day")
            print(f"=====================================")
        else:
            print(
                f"  current episode's total training time: {current_session_when:.2f}sec / {current_session_when / 60:.2f}minutes / {current_session_when / 3600:.2f}Smallwhen / {current_session_when / 86400:.2f}Day")
            print(
                f"  cumulative total training time: {total_training_when:.2f}sec /  {total_training_when / 60:.2f}minutes / {total_training_when / 3600:.2f}Smallwhen / {total_training_when / 86400:.2f}Day")
            print(f"=====================================")
        # ===== recordTestmetricsto TensorBoard =====
        final_test_results = test_results
        writer.add_scalar("test/meta_loss", test_results.get("meta_loss", 0.0), ep)
        save_family_metrics(
            [{"family": fam, **fm} for fam, fm in test_results.get("family_metrics", {}).items()],
            os.path.join(metrics_dir, "test_family_metrics.csv")
        )
        writer.add_scalar("test/acc", test_results['acc'], ep)
        writer.add_scalar("test/recall", test_results['recall'], ep)
        writer.add_scalar("test/f1", test_results['f1'], ep)
        # ======================================
        save_family_summary(
            test_results.get("family_metrics", {}),
            os.path.join(metrics_dir, "test_family_metrics_B.csv"),
        )
        plot_eval_artifacts(
            test_results.get("probabilities", []),
            test_results.get("labels", []),
            test_results.get("predictions", []),
            metrics_dir,
            "test",
        )

    if final_val_results is not None or final_test_results is not None:
        print(f"[Exp-B][Step9] Evaluation completed, newly added figures with family metrics output to {metrics_dir}/")

    # ===== Close TensorBoard logger =====
    writer.close()

    # for eventually trained models in-line sparsity evaluation
    report_sparsity(model, threshold=3.0, save_dir=metrics_dir)

    return model, (train_tasks, val_tasks, test_tasks), (normal_train, normal_val, normal_test)


# ===== EvaluationFunctionwithTrainingIdenticalfunctional inner update=====
def get_rmin_linear(current_ep: int,
                    total_eps: int,
                    r_start: float,
                    r_target: float,
                    warmup_frac: float) -> float:
    """
    linear threshold increase adjustment function: based on current training progress, at linear region between will R_min from r_start gradually step increase to r_targetet

    parameterNOTE
    - current_ep: CurrentTraininging episodesepisode1 Starting
    - total_eps: planned total training episodesForCalculateprogress
    - r_start:   recall floorofstarting value
    - r_target:  recall flooroftargetvalue
    - warmup_frac: threshold increase proportion of training progress ratio (0~1), For example 0.4 indicates before 40% training progress in linear line increase to r_target

    Return
    - current epoch should use R_min value
    """
    total_eps = max(1, int(total_eps))
    warmup_eps = max(1, int(total_eps * max(0.0, min(1.0, warmup_frac))))
    # linear interpolation, after warmup region fixed for r_target
    progress = min(max(current_ep, 0), warmup_eps)
    r_min = r_start + (r_target - r_start) * (progress / float(warmup_eps))
    return float(min(r_target, max(r_start, r_min)))


def get_rmin_ladder(recall_history: list,
                    r_current: float,
                    r_target: float,
                    window: int,
                    margin: float,
                    step: float) -> float:
    """
    adaptive ladder threshold increasereservedtime recent window when validation's Recall@t* mean >= r_current + margin then
    will R_min on adjustment stepnot exceeding r_targetotherwise keep unchanged

    parameterNOTE
    - recall_history: recent several times validation's Recall@t* sequencebased on time increasing
    - r_current: Current R_min
    - r_target: target maximum R_min
    - window: observation window size
    - margin: adjustment determination margin
    - step: per time adjustment amplitude
    """
    if window <= 0 or len(recall_history) < window:
        return float(r_current)
    window_vals = recall_history[-window:]
    if min(window_vals) >= (r_current + margin):
        return float(min(r_target, r_current + step))
    return float(r_current)


def select_threshold_with_recall_floor(y_true,
                                       y_prob,
                                       r_min: float = 0.60,
                                       beta: float = 2.0,
                                       prev_thr: float = None,
                                       ema: float = 0.2,
                                       pos_rate_cap: float = 0.90,
                                       precision_min: float = 0.40,
                                       enforce_pos_rate_cap: bool = True) -> dict:
    """
    Based on“recall floorconstraint”of threshold valueSelect
    1) exhaustively try one group of threshold valuesfilter Recall >= r_min and Precision >= precision_min candidate set
       If enforce_pos_rate_cap=True, also require positive class prediction rate pos_rate <= pos_rate_cap
    2) from candidates select F_beta maximum threshold value as t*
       If infeasible (no threshold value satisfy), then fallback to select“Recall highest”of threshold value
    3) for selected threshold value do EMA smoothing (if provide prev_thr), reduce fluctuations between adjacent evaluations
    4) Return t* and at t* under Precision/Recall/F_beta/positive class prediction rate/feasibility

    NOTE
    - y_true should be binary labels (0/1); y_prob should be positive class probabilities (if logits, apply sigmoid
    - This function is only for validation/selection phase, will not affect backpropagation in training process.
    """
    import numpy as np

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if y_true.size == 0 or y_prob.size == 0:
        # Fallback return for empty input
        return {
            "threshold": float(prev_thr if prev_thr is not None else 0.5),
            "precision": 0.0,
            "recall": 0.0,
            "fbeta": 0.0,
            "pos_rate": 0.0,
            "feasible": False,
        }

    # Construct candidate threshold values (including endpoints), appropriate quantity to avoid slowness
    thresholds = np.linspace(0.0, 1.0, 101)
    beta2 = float(beta * beta)

    best_idx = None
    best_fbeta = -1.0
    feasible_any = False

    # In order to fallback: record“highest recall”of threshold value
    top_recall = -1.0
    top_recall_idx = None
    # Record highest recall under pos_rate constraint (if necessary)
    top_recall_under_cap = -1.0
    top_recall_under_cap_idx = None

    for i, th in enumerate(thresholds):
        y_pred = (y_prob >= th).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        pos_rate = float(y_pred.mean())
        if recall > top_recall:
            top_recall = recall
            top_recall_idx = i
        # Record highest recall under positive rate upper limit constraint (if enabled)
        if (not enforce_pos_rate_cap) or (pos_rate <= pos_rate_cap):
            if recall > top_recall_under_cap:
                top_recall_under_cap = recall
                top_recall_under_cap_idx = i

        denom = (beta2 * precision + recall)
        fbeta = (1.0 + beta2) * precision * recall / max(denom, 1e-12)

        # Feasibility: simultaneously satisfy recall floor, precision floor, and (optional) positive rate upper limit
        if (recall >= r_min) and (precision >= precision_min) and (
                (not enforce_pos_rate_cap) or (pos_rate <= pos_rate_cap)
        ):
            feasible_any = True
            if fbeta > best_fbeta:
                best_fbeta = fbeta
                best_idx = i

    if feasible_any:
        raw_thr = float(thresholds[best_idx])
    else:
        # infeasible: preferentially select“under positive rate constraint recall highest”of threshold valuesecondly fallback to absolute highest recall
        if top_recall_under_cap_idx is not None:
            raw_thr = float(thresholds[top_recall_under_cap_idx])
        else:
            raw_thr = float(thresholds[top_recall_idx])

    # EMA smoothing of threshold value to reduce jitter
    if prev_thr is not None:
        selected_thr = float((1.0 - ema) * float(prev_thr) + ema * raw_thr)
    else:
        selected_thr = raw_thr

    y_pred_sel = (y_prob >= selected_thr).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred_sel == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred_sel == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred_sel == 0)))
    precision_sel = tp / max(tp + fp, 1)
    recall_sel = tp / max(tp + fn, 1)
    denom = (beta2 * precision_sel + recall_sel)
    fbeta_sel = (1.0 + beta2) * precision_sel * recall_sel / max(denom, 1e-12)
    pos_rate_sel = float(y_pred_sel.mean())

    return {
        "threshold": selected_thr,
        "precision": float(precision_sel),
        "recall": float(recall_sel),
        "fbeta": float(fbeta_sel),
        "pos_rate": float(pos_rate_sel),
        "feasible": bool(feasible_any),
    }


def evaluate_on_split(
        model,
        task_dict_split,
        normal_pool_split,
        episodes=50,
        device=DEVICE,
        lr_inner=lr_inner,
        inner_steps=inner_steps,
        first_order=False,
        # === Insert point B: newly added sanity switch ===
        shuffle_labels=False,
        no_adapt=False,
        K_override=None,
        inner_steps_override=None,
        mode="one_vs_rest",
        incident_df_all=None,  #  Must pass in
        show_prob_hist=False,
        search_best_threshold=False,
        # ===== sampling with threshold value control =====
        hard_neg: bool = False,
        other_abnormal_ratio: float = OTHER_ABNORMAL_RATIO,
        fixed_threshold: optional[float] = None,
        # ===== Fig.2/Fig.4 Data Export (2026-03-03 ) =====
        save_scores: bool = False,  # Whether to save y_true/y_scores to CSV (for Fig.2 PR Curve)
        method_name: str = "unknown",  # Method name (for file naming)
        output_dir: str = "outputs/fig2",  # OutputDirectory
        seed: int = 42,  # Random seed (for file naming, 2026-03-03 )
):
    # Optional: sanity check
    assert incident_df_all is not None, "evaluate_on_split requires incident_df_allentire event summary table"

    model.eval()
    #  Fix: select loss function based on model type (adapt to JointTraininging/LogReg single logit outputut
    if USE_COMPARISON_MODEL:
        # Comparison models (JointTraininging/LogReg) output [N] single logit, use BCEWithLogitsLoss
        # NOTE: reduction must be 'mean', otherwise torch.autograd.grad will error (requires scalar outputut
        loss_fn = nn.BCEwithLogitsLoss(reduction="mean")
    else:
        # SABLE models output [N, 2] binary class logits, use CrossEntropyLoss
        loss_fn = nn.CrossEntropyLoss()
    accs = []
    recalls = []
    aucs = []
    pr_aucs = []
    f1s = []
    meta_losses = []  # Record cross-entropy loss on query set
    all_probs = []  # Collect all prediction probabilities for histogram
    all_preds = []  # Collect all prediction labels for decision
    all_labels = []  # Collect all ground truth labels for threshold search
    fam_preds = defaultdict(list)  # Family-level predictions
    fam_labels = defaultdict(list)
    fam_probs = defaultdict(list)  # ===== family-level probabilities (for Fig.4 AUC Calculate=====
    # ===== episode-level AUC collection (for Fig.4, 2026-03-03 ) =====
    episode_auc_records = []  # Format: [episode_id, family_id, roc_auc, pr_auc, n_query, pos_rate]

    # If override inner_steps
    inner_steps_eval = inner_steps if inner_steps_override is None else inner_steps_override

    for episode_idx in range(episodes):
        # K=1/other override
        k_use = K_override if K_override is not None else K_SHOT

        # Key: incident_df_all passed to sample_task
        try:
            sample_result = sample_task(
                task_dict=task_dict_split,  # use current split of task dictionaryavoid cross-dataset sampling
                current_split_tasks=task_dict_split,
                normal_pool=normal_pool_split,
                k_shot=k_use, q_shot=Q_SHOT,
                mode=mode,
                incident_df_all=incident_df_all,
                debug=False,
                other_abnormal_ratio=other_abnormal_ratio,  # keepconsistentofratio
                hard_neg=hard_neg,
            )
            sx, sy, qx, qy, _info = _unwrap_sample_task(sample_result)
        except RunwhenError:
            # Skipped this one episode
            continue

        sx, sy, qx, qy = sx.to(device), sy.to(device), qx.to(device), qy.to(device)

        # Label shufflesanity
        if shuffle_labels:
            perm = torch.randperm(qy.size(0), device=qy.device)
            qy = qy[perm]

        # functional inner-loop
        fast = model.named_params_dict()
        if not no_adapt:
            for _step in range(inner_steps_eval):
                # at adaptation phase, simulate training time's noisy environment
                # Standardmodelsofforward_with_weightsNo is_training parameter
                logits_s = model.forward_with_weights(sx, fast)
                #  fixbased onmodelsTypeProcessOutputFormatJointTraininging/LogRegCompatible with
                if USE_COMPARISON_MODEL and COMPARISON_MODEL_NAME in ["JointTraininging", "LogReg"]:
                    # JointTraining/LogReg Output[N] shape, directly use
                    loss_s = loss_fn(logits_s, sy.float())
                elif USE_COMPARISON_MODEL:
                    # Other comparison models Output[N,2] dimension, take positive scoreslogits[:,1]
                    loss_s = loss_fn(logits_s[:, 1], sy.float())
                else:
                    # CrossEntropyLoss expects Long type target
                    loss_s = loss_fn(logits_s, sy)
                grads = torch.autograd.grad(
                    loss_s, list(fast.values()),
                    create_graph=not first_order,
                    retain_graph=not first_order,
                    allow_unused=True
                )
                # Filter out None gradients
                new_fast = OrderedDict()
                # ===== note: In order to code concisehere can use one small helper or directly write =====
                grad_idx = 0
                for n, w in fast.items():
                    if w.requires_grad:
                        g = grads[grad_idx]
                        grad_idx += 1
                        if g is not None:
                            new_fast[n] = w - lr_inner * g
                        else:
                            new_fast[n] = w
                    else:
                        new_fast[n] = w
                fast = new_fast

        with torch.no_grad():
            q_logits = model.forward_with_weights(qx, fast)
            # Calculatequery setoflosswithPredictionbased onmodelsTypeProcesstarget
            if USE_COMPARISON_MODEL and COMPARISON_MODEL_NAME in ["JointTraininging", "LogReg"]:
                # JointTraining/LogReg Output[N] shape, directly use
                loss_q = loss_fn(q_logits, qy.float()).item()
            elif USE_COMPARISON_MODEL:
                # Other comparison models Output[N,2] dimension, take positive scoresq_logits[:,1]
                loss_q = loss_fn(q_logits[:, 1], qy.float()).item()
            else:
                # CrossEntropyLoss expects Long type target
                loss_q = loss_fn(q_logits, qy).item()
            meta_losses.append(loss_q)
            #  fixbased onmodelsOutputFormatCalculateprobabilityAndPredictionJointTraininging/LogRegCompatible with
            if q_logits.ndim == 2 and q_logits.shape[1] == 2:
                # Dual-class logits Output[N, 2]
                probs = torch.softmax(q_logits, dim=1)[:, 1].cpu().numpy()
                pred_tensor = q_logits.argmax(1)
            else:
                # Single logit Output[N]JointTraininging/LogReg
                probs = torch.sigmoid(q_logits.squeeze(-1)).cpu().numpy()
                pred_tensor = (q_logits.squeeze(-1) > 0).long()
            preds = pred_tensor.cpu().numpy()
            # Determination: If provide fixed threshold then based on threshold value determineotherwise use on-the-fly prediction
            if fixed_threshold is not None:
                preds = (probs >= float(fixed_threshold)).astype(np.int64)
                pred_tensor = torch.from_numpy(preds).to(qy.device)
            acc = (pred_tensor == qy).float().mean().item()
            accs.append(acc)
            labels = qy.cpu().numpy()
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            # based on family recordPredictionResults
            fam = _info.get("family", "unknown")
            fam_preds[fam].extend(preds.tolist())
            fam_labels[fam].extend(labels.tolist())
            fam_probs[fam].extend(probs.tolist())  # ===== collect family level probability =====

            # ===== collect episode-level AUC(For Fig.4, 2026-03-03 )=====
            n_query = len(labels)
            pos_rate = float(np.mean(labels))
            try:
                ep_roc_auc = roc_auc_score(labels, probs)
            except Exception:
                ep_roc_auc = 0.5
            try:
                ep_pr_auc = average_precision_score(labels, probs)
            except Exception:
                ep_pr_auc = 0.5

            episode_auc_records.append({
                'episode_id': episode_idx,
                'family_id': fam,
                'roc_auc': float(ep_roc_auc),
                'pr_auc': float(ep_pr_auc),
                'n_query': n_query,
                'pos_rate': pos_rate
            })

            # Calculate each class metrics
            try:
                aucs.append(ep_roc_auc)  # use calculated value
            except Exception:
                aucs.append(0.5)
            try:
                pr_aucs.append(ep_pr_auc)  # use calculated value
            except Exception:
                pr_aucs.append(0.5)
            try:
                f1s.append(f1_score(labels, preds))
            except Exception:
                f1s.append(0.0)
            try:
                recalls.append(recall_score(labels, preds))
            except Exception:
                recalls.append(0.0)

    model.train()
    results = {
        "meta_loss": float(np.mean(meta_losses)) if meta_losses else 0.0,
        "acc": float(np.mean(accs)) if accs else 0.5,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "roc_auc": float(np.mean(aucs)) if aucs else 0.5,
        "pr_auc": float(np.mean(pr_aucs)) if pr_aucs else 0.5,
        "f1": float(np.mean(f1s)) if f1s else 0.0,
    }

    # recordOriginalprobabilitywithLabelfacilitatesubsequentplotCurve
    results["probabilities"] = all_probs
    results["labels"] = all_labels
    results["predictions"] = all_preds

    # probability histogram with judgment threshold value check
    if all_probs:
        hist, bin_edges = np.histogram(all_probs, bins=10, range=(0, 1))
        results["prob_hist"] = {
            "hist": hist.tolist(),
            "bins": bin_edges.tolist(),
        }
        if show_prob_hist:
            print(f"  Probability histogram: {hist.tolist()}")
        pred_rate = float(np.mean(all_preds)) if all_preds else 0.0
        results["pos_pred_rate"] = pred_rate
        if pred_rate > 0.95:
            print("  WARNING: Model almost always predicts positive class, consider adjusting threshold or using weighted loss")
        elif pred_rate < 0.05:
            print("  WARNING: Model almost always predicts negative class, consider adjusting threshold or using weighted loss")

        # Optional: search for optimal threshold value on validation set
        if search_best_threshold and all_labels:
            thresholds = np.linspace(0.05, 0.95, 19)
            labels_arr = np.array(all_labels)
            best_th, best_f1 = 0.5, 0.0
            for th in thresholds:
                pred_tmp = (np.array(all_probs) >= th).astype(int)
                try:
                    f1_tmp = f1_score(labels_arr, pred_tmp)
                except Exception:
                    f1_tmp = 0.0
                if f1_tmp > best_f1:
                    best_f1 = f1_tmp
                    best_th = th
            results["best_threshold"] = float(best_th)
            results["best_f1"] = float(best_f1)

    # Calculate family-level metrics (===== 
    family_metrics = {}
    for fam, preds_list in fam_preds.items():
        labels = np.array(fam_labels[fam])
        preds_arr = np.array(preds_list)
        probs_arr = np.array(fam_probs[fam])  # Getprobabilityarray
        acc = (preds_arr == labels).mean() if len(labels) else 0.0
        try:
            rec = recall_score(labels, preds_arr)
        except Exception:
            rec = 0.0
        try:
            f1 = f1_score(labels, preds_arr)
        except Exception:
            f1 = 0.0
        # ===== calculate family-level ROC-AUC and PR-AUC =====
        # 
        unique_labels = np.unique(labels)
        is_single_class = len(unique_labels) < 2

        if is_single_class:
            # Single-class label: AUC is meaningless, record as NaN
            fam_roc_auc = np.nan
            fam_pr_auc = np.nan
        else:
            try:
                fam_roc_auc = roc_auc_score(labels, probs_arr)
            except Exception:
                fam_roc_auc = 0.5
            try:
                fam_pr_auc = average_precision_score(labels, probs_arr)
            except Exception:
                fam_pr_auc = 0.5
        family_metrics[fam] = {
            "acc": float(acc),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(fam_roc_auc),  # newly added
            "pr_auc": float(fam_pr_auc)      # newly added
        }
    results["family_metrics"] = family_metrics

    # ===== newly addedSave y_true/y_scores to CSVFor Fig.2 PR Curve=====
    # 
    if save_scores and len(all_probs) > 0 and len(all_labels) > 0:
        import pandas as pd
        from datewhen import datewhen
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename containingconfiguration info (prevent overwriting)
        date_str = datewhen.now().strfwhen("%Y%m%d")
        filename = f"{method_name}_ep{episodes}_seed{seed}_{date_str}.csv"
        scores_path = os.path.join(output_dir, filename)

        scores_df = pd.DataFrame({
            'y_true': all_labels,
            'y_score': all_probs
        })
        scores_df.to_csv(scores_path, index=False)
        print(f"[OK] saved {method_name} PR curve data to: {scores_path}")
        print(f"  configuration: episodes={episodes}, seed={seed}, sample count={len(all_labels)}")

        # ===== save Fig.4 two independent files (2026-03-03 ) =====
        if len(episode_auc_records) > 0:
            # create Fig.4 OutputDirectory
            fig4_dir = "outputs/fig4"
            os.makedirs(fig4_dir, exist_ok=True)

            # 1) Save episode-level file
            episode_df = pd.DataFrame(episode_auc_records)
            episode_filename = f"episode_auc_{method_name}_ep{episodes}_seed{seed}_{date_str}.csv"
            episode_path = os.path.join(fig4_dir, episode_filename)
            episode_df.to_csv(episode_path, index=False)
            print(f"[OK] saved Fig.4 Episode-level AUC to: {episode_path}")
            print(f"   Episode record count: {len(episode_auc_records)}")

            # 2) Save family-level file (aggregated from episode records)
            # 
            family_agg_records = []
            skipped_single_class_count = 0
            for fam in family_metrics.keys():
                # Collect all episode records for this family
                fam_episodes = [ep for ep in episode_auc_records if ep['family_id'] == fam]
                if len(fam_episodes) == 0:
                    continue  # SkippedNo episode Dataof family

                # Filter out NaN values (invalid AUC caused by single-class label)
                fam_roc_auc_valid = [ep['roc_auc'] for ep in fam_episodes if not np.isnan(ep['roc_auc'])]
                fam_pr_auc_valid = [ep['pr_auc'] for ep in fam_episodes if not np.isnan(ep['pr_auc'])]

                # Keep only families with at least one valid AUC
                if len(fam_roc_auc_valid) > 0 or len(fam_pr_auc_valid) > 0:
                    family_agg_records.append({
                        'method': method_name,
                        'family_id': fam,
                        'roc_auc_mean': float(np.mean(fam_roc_auc_valid)) if fam_roc_auc_valid else np.nan,
                        'roc_auc_std': float(np.std(fam_roc_auc_valid)) if len(fam_roc_auc_valid) > 1 else 0.0,
                        'pr_auc_mean': float(np.mean(fam_pr_auc_valid)) if fam_pr_auc_valid else np.nan,
                        'pr_auc_std': float(np.std(fam_pr_auc_valid)) if len(fam_pr_auc_valid) > 1 else 0.0,
                        'num_episodes': len(fam_episodes)
                    })
                else:
                    skipped_single_class_count += 1

            family_df = pd.DataFrame(family_agg_records)
            family_filename = f"family_auc_{method_name}_ep{episodes}_seed{seed}_{date_str}.csv"
            family_path = os.path.join(fig4_dir, family_filename)
            family_df.to_csv(family_path, index=False)
            print(f"[OK] saved Fig.4 Family-level AUC to: {family_path}")
            print(f"   Family quantity: {len(family_agg_records)}")
            # PrintstatisticsINFOSkippedof family quantity
            valid_families = len(family_agg_records)
            total_families_in_records = len(set(ep['family_id'] for ep in episode_auc_records))
            print(f"   Valid Family (AUC Success): {valid_families}")
            print(f"   Total family count: {total_families_in_records}")
            if skipped_single_class_count > 0:
                print(f"    Skipped family count: {skipped_single_class_count} (single-class label, AUC meaningless)")
            elif valid_families < total_families_in_records:
                print(f"    Skipped family count: {total_families_in_records - valid_families} (no valid episodes)")

    return results


# =========================Check Function Area==============================
def check_label_leakage(task_dict, incident_df, normal_pool):
    """Check if label leakage exists"""
    print("\n=== Label Leakage Check ===")

    # 1. Check feature dimension consistency
    feature_dims = set()
    for task, pairs in task_dict.items():
        for iid, sid in pairs:
            feat = extract_abnormal_features(incident_df, iid, sid)
            if feat is not None:
                feature_dims.add(feat.shape[0])

    normal_dims = set()
    for norm_feat in normal_pool[:100]:  # checkbefore100
        if norm_feat is not None:
            normal_dims.add(len(norm_feat))

    print(f"exceptionFeature dimensions: {feature_dims}")
    print(f"Normal feature dimensions: {normal_dims}")

    if len(feature_dims) > 1 or len(normal_dims) > 1:
        print("  WARNING: Feature dimensions inconsistent!")

    # 2. checkFeaturevaluedistributiondifference
    abnormal_samples = []
    for task, pairs in list(task_dict.items())[:5]:  # checkbefore5Task
        for iid, sid in pairs[:3]:  # Each task take 3 samples
            feat = extract_abnormal_features(incident_df, iid, sid)
            if feat is not None:
                abnormal_samples.append(feat)

    normal_samples = [f for f in normal_pool[:50] if f is not None]

    if abnormal_samples and normal_samples:
        abnormal_mean = np.mean([np.mean(s) for s in abnormal_samples])
        normal_mean = np.mean([np.mean(s) for s in normal_samples])
        abnormal_std = np.mean([np.std(s) for s in abnormal_samples])
        normal_std = np.mean([np.std(s) for s in normal_samples])

        print(f"abnormal sample statistics: mean value={abnormal_mean:.4f}, Standard deviation={abnormal_std:.4f}")
        print(f"normal samples statistics: mean value={normal_mean:.4f}, Standard deviation={normal_std:.4f}")

        mean_diff = abs(abnormal_mean - normal_mean)
        if mean_diff > 2.0:  # threshold value adjustable
            print(f"  WARNING: abnormal/normal samples mean value difference too large ({mean_diff:.4f})")

    # 3. checkwhenleakage
    if 'Week' in incident_df.columns:
        week_distribution = incident_df['Week'].value_counts()
        print(f"Weekdistribution: {dict(week_distribution.head())}")
        if len(week_distribution) < 3:
            print("  WARNING: Week distribution too concentrated, may exist time leakage")

    print("==================\n")


# ===== Split overlap check (Based on positive (IncidentId, SensorNumber)=====
def check_split_overlap_stats(train_tasks, val_tasks, test_tasks):
    """
    visible train/val/test of (IncidentId, SensorNumber) if there is intersection (only positive pool, negative from normal pools not check here).
    """

    def flatten_pairs(d):
        s = set()
        for k, v in d.items():
            for pair in v:
                s.add(tuple(pair))
        return s

    s_train = flatten_pairs(train_tasks)
    s_val = flatten_pairs(val_tasks)
    s_test = flatten_pairs(test_tasks)

    inter_tv = len(s_train & s_val)
    inter_tt = len(s_train & s_test)
    inter_vt = len(s_val & s_test)

    print("\n   Split overlap check (Based on positive (IncidentId,SensorNumber)")
    print(f"  • |train∩val|  = {inter_tv}")
    print(f"  • |train∩test| = {inter_tt}")
    print(f"  • |val∩test|   = {inter_vt}")
    if inter_tv == inter_tt == inter_vt == 0:
        print("No intersection (appears safe)")
    else:
        print("Detected cross-split overlap, please check task construction / split logic")


# ===== Sanity Suitequick check=====
def run_sanity_suite(model, splits, normal_pools, device=DEVICE, episodes=50, mode="one_vs_rest",
                     incident_df_all=None):
    """
    splits: (train_tasks, val_tasks, test_tasks)
    normal_pools: (normal_train, normal_val, normal_test)
    """
    train_tasks, val_tasks, test_tasks = splits
    normal_train, normal_val, normal_test = normal_pools

    print("\n================ SANITY SUITE (on VAL) ================")

    # 1) Label Shuffle (should be ≈ 0.5)
    acc_ls_results = evaluate_on_split(
        model, val_tasks, normal_val,
        episodes=episodes, device=device, shuffle_labels=True,
        mode=mode, incident_df_all=incident_df_all
    )
    print(f"Label Shuffle Acc ≈ {acc_ls_results['acc']:.3f}  (expected ≈ 0.5)")

    # 2) 0-step baseline
    acc_0_results = evaluate_on_split(
        model, val_tasks, normal_val,
        episodes=episodes, device=device,
        no_adapt=True, mode=mode, incident_df_all=incident_df_all
    )
    print(f"0-step Acc        = {acc_0_results['acc']:.3f}, Recall = {acc_0_results['recall']:.3f}, "
          f"ROC-AUC = {acc_0_results['roc_auc']:.3f}")

    # 3) K=1
    acc_k1_results = evaluate_on_split(
        model, val_tasks, normal_val,
        episodes=episodes, device=device,
        K_override=1, inner_steps_override=1,
        mode=mode, incident_df_all=incident_df_all
    )
    print(f"K=1 Acc           = {acc_k1_results['acc']:.3f}, Recall = {acc_k1_results['recall']:.3f}, "
          f"ROC-AUC = {acc_k1_results['roc_auc']:.3f}")

    # 4) inner-steps scan
    for s in (0, 1, 3, 5):
        acc_s_results = evaluate_on_split(
            model, val_tasks, normal_val,
            episodes=episodes, device=device,
            inner_steps_override=s, mode=mode, incident_df_all=incident_df_all
        )
        print(f"inner_steps={s:<2}   Acc = {acc_s_results['acc']:.3f}, "
              f"Recall = {acc_s_results['recall']:.3f}, ROC-AUC = {acc_s_results['roc_auc']:.3f}")

    print("========================================================\n")


# =========================Check Function Area==============================


# ===========================Utility Area===============================
# ===== Model Factory =====
# Unified management of available baseline models, facilitating subsequent extension or troubleshooting configuration problems
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "mlp": StandardMLP,
    "cnn1d": CNN1D,
    "lstm": LSTM,
    "transformer": StandardTransformer,
    "b_transformer": B_SMALL_Transformer,
}


def get_model(model_type: str = "mlp", input_dim: int = FEATURE_DIM, model_kwargs: optional[Dict] = None) -> BaseModel:
    """Return corresponding baseline model instance based on model type."""
    if model_kwargs is None:
        model_kwargs = {}
    else:
        model_kwargs = dict(model_kwargs)

    model_type_key = model_type.strip().lower()
    if model_type_key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise valueError(f"unknownofmodelsType: {model_type}optionalvalue: {available}")

    model_cls = MODEL_REGISTRY[model_type_key]
    try:
        return model_cls(input_dim=input_dim, **model_kwargs)
    except TypeError as exc:
        raise TypeError(f"buildmodels {model_type} FailedPlease checkhyperparameterconfiguration: {exc}") from exc


def create_model(model_type="mlp", input_dim=FEATURE_DIM, l2_reg=1e-4, **kwargs):
    """Compatible with legacy version adjustment use of factory wrapper, subsequently recommended to directly use get_model."""
    model_kwargs = dict(kwargs)
    if model_type.lower() == "mlp" and "l2_reg" not in model_kwargs:
        model_kwargs["l2_reg"] = l2_reg
    return get_model(model_type=model_type, input_dim=input_dim, model_kwargs=model_kwargs)


# ===== KL Divergence Beta Warmup Scheduler =====
class BetaScheduler:
    """
    LinearWarmup: Adjuster for Beta value during warmup.
    Beta linearly increases from 0 to 1 over specified warmup_steps, then stays at 1.
    """

    def __init__(self, warmup_steps):
        self.warmup_steps = warmup_steps
        self.slope = 1.0 / self.warmup_steps if self.warmup_steps > 0 else 1.0

    def get_beta(self, current_step):
        """
        Get corresponding beta value based on current training step.
        """
        if self.warmup_steps == 0:
            return 1.0
        if current_step < self.warmup_steps:
            return self.slope * current_step
        return 1.0


# ===== B-SMALL Model Sparsity Evaluation =====
def report_sparsity(model: nn.Module, threshold: float = 3.0, save_dir: str = "metrics"):
    """
    Calculate and print sparsity of a B-SMALL model, simultaneously generate visualization plots and data report

    Sparsity is defined as the percentage of 'pruned' weights to total weights in the model.
    A weight is considered 'pruned' if its corresponding log_alpha > threshold.

    parameters:
        model (nn.Module): Traininged B-SMALL model.
        threshold (float): Log_alpha threshold value to decide if weight is pruned.
                           SVD paper suggested value is 3.0.
        save_dir (str): SaveFiguresAndReportofDirectoryDefault is "metrics"

    Generatefile:
        - sparsity_by_layer.png: sparsity bar chart for each layer
        - log_alpha_distribution.png: histogram of log_alpha values
        - sparsity_pattern.png: heatmap of weight sparsity patterns
        - sparsity_report.csv: DetailedDataReport
    """
    total_weights = 0
    pruned_weights = 0

    # Collect detailed info for each layer
    layer_stats = []
    all_log_alphas = []
    layer_log_alphas = {}

    # Ensure model is in evaluation mode
    model.eval()

    # Iterate through all submodules of the model
    layer_idx = 0
    for name, module in model.named_modules():
        # Check if this is our custom sparse layer
        if isinstance(module, SparseVDLinear):
            # Get log_alpha values for all weights in this layer
            log_alpha = module.log_alpha.detach().cpu().numpy()

            # Basic layer information
            layer_name = name if name else f"SparseVDLinear_{layer_idx}"
            in_features = module.in_features
            out_features = module.out_features
            layer_total = module.theta.nelement()
            layer_pruned = (module.log_alpha > threshold).sum().item()
            layer_sparsity = 100.0 * layer_pruned / layer_total if layer_total > 0 else 0.0

            # recordstatisticsINFO
            layer_stats.append({
                'layer_name': layer_name,
                'in_features': in_features,
                'out_features': out_features,
                'total_weights': layer_total,
                'pruned_weights': layer_pruned,
                'sparsity_percent': layer_sparsity
            })

            # Collect log_alpha for global distribution
            all_log_alphas.extend(log_alpha.flatten().tolist())
            layer_log_alphas[layer_name] = log_alpha

            # Accumulate total weight count
            total_weights += layer_total
            pruned_weights += layer_pruned

            layer_idx += 1

    if total_weights == 0:
        print("No SparseVDLinear layer found in model, cannot calculate sparsity.")
        return 0.0

    sparsity_percentage = 100.0 * pruned_weights / total_weights

    # Print console report (keep original functionality)
    print(f"\n========== Model Sparsity Report =========="
    print(f"pruningthreshold value (log_alpha): > {threshold}")
    print(f"Total learnable weights: {total_weights}")
    print(f"pruned byweight: {pruned_weights}")
    print(f"Model sparsity: {sparsity_percentage:.2f}%")
    print(f"discover {len(layer_stats)}  SparseVDLinear layer")
    print(f"=====================================")

    # createSaveDirectory
    os.makedirs(save_dir, exist_ok=True)

    # ===== 1. SaveDetailedDataCSV =====
    if layer_stats:
        df = pd.DataFrame(layer_stats)
        # Add summary line
        summary_row = {
            'layer_name': 'TOTAL',
            'in_features': '-',
            'out_features': '-',
            'total_weights': total_weights,
            'pruned_weights': pruned_weights,
            'sparsity_percent': sparsity_percentage
        }
        df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

        csv_path = os.path.join(save_dir, "sparsity_report.csv")
        df.to_csv(csv_path, index=False)
        print(f" Sparsity detailed report saved: {csv_path}")

    # ===== 2. Plot layer-wise sparsity bar chart =====
    if layer_stats:
        plt.figure(figsize=(10, 6))
        layer_names = [s['layer_name'] for s in layer_stats]
        sparsities = [s['sparsity_percent'] for s in layer_stats]

        bars = plt.bar(range(len(layer_names)), sparsities, color='steelblue', alpha=0.7)
        plt.axhline(y=sparsity_percentage, color='red', linestyle='--',
                    label=f'Overall Sparsity: {sparsity_percentage:.2f}%')

        plt.xlabel('Layer', fontsize=12)
        plt.ylabel('Sparsity (%)', fontsize=12)
        plt.title(f'Sparsity by Layer (Threshold: log_alpha > {threshold})', fontsize=14)
        plt.xticks(range(len(layer_names)), layer_names, rotation=45, ha='right')
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        bar_path = os.path.join(save_dir, "sparsity_by_layer.png")
        plt.savefig(bar_path, dpi=150)
        plt.close()
        print(f" Layer-wise sparsity bar chart saved: {bar_path}")

    # ===== 3. plotlog_alphadistributionhistogram =====
    if all_log_alphas:
        plt.figure(figsize=(10, 6))

        # plothistogram
        n, bins, patches = plt.hist(all_log_alphas, bins=50, color='skyblue',
                                    edgecolor='black', alpha=0.7)

        # Mark pruning threshold line
        plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                    label=f'Pruning Threshold: {threshold}')

        # Fill pruning area
        for i, patch in enumerate(patches):
            if bins[i] > threshold:
                patch.set_facecolor('salmon')
                patch.set_alpha(0.7)

        plt.xlabel('log_alpha', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of log_alpha values', fontsize=14)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        hist_path = os.path.join(save_dir, "log_alpha_distribution.png")
        plt.savefig(hist_path, dpi=150)
        plt.close()
        print(f" log_alpha distribution histogram saved: {hist_path}")

    # ===== 4. Plot weight sparsity pattern heatmap =====
    if layer_log_alphas:
        num_layers = len(layer_log_alphas)
        fig, axes = plt.subplots(1, min(num_layers, 4), figsize=(min(num_layers * 4, 16), 4))

        # Process single layer case
        if num_layers == 1:
            axes = [axes]

        for idx, (layer_name, log_alpha) in enumerate(list(layer_log_alphas.items())[:4]):
            ax = axes[idx] if num_layers > 1 else axes[0]

            # Create sparsity pattern matrix (1=pruned, 0=kept)
            sparsity_mask = (log_alpha > threshold).astype(float)

            # plotheatmap
            im = ax.imshow(sparsity_mask, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
            ax.set_title(f'{layer_name}\n({layer_stats[idx]["sparsity_percent"]:.1f}% pruned)',
                         fontsize=10)
            ax.set_xlabel('Input Dim', fontsize=9)
            ax.set_ylabel('Output Dim', fontsize=9)

            # Add color items
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(['Kept', 'Pruned'])

        # Hidden many subplots
        if num_layers < 4:
            for idx in range(num_layers, min(4, len(axes))):
                axes[idx].axis('off')

        plt.suptitle(f'Weight Sparsity Pattern (Red=Pruned, Green=Kept)', fontsize=14, y=1.02)
        plt.tight_layout()

        pattern_path = os.path.join(save_dir, "sparsity_pattern.png")
        plt.savefig(pattern_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f" weight sparsity pattern heatmap saved: {pattern_path}")

    print(f"All sparsity visualization files saved to: {save_dir}/")

    return sparsity_percentage


# ===== Cosine Similarity CalculationForEvaluation=====
def _cosine_sim(g1, g2, eps=1e-8):
    num = 0.0
    den1 = 0.0
    den2 = 0.0
    for a, b in zip(g1, g2):
        va = a.reshape(-1)
        vb = b.reshape(-1)
        num += (va * vb).sum()
        den1 += (va * va).sum()
        den2 += (vb * vb).sum()
    return num / (den1.sqrt() * den2.sqrt() + eps)


# # ===== cosine similarity logits calculationForEvaluation=====
def cosine_logits(emb_q, proto_pos, proto_neg, scale=10.0):
    # emb/proto are already same-dimensional vectors, do L2 normalization
    def _norm(x): return x / (x.norm(dim=1, keepdim=True) + 1e-8)

    q = _norm(emb_q)
    p1 = _norm(proto_pos.unsqueeze(0))  # [1,D]
    p0 = _norm(proto_neg.unsqueeze(0))
    logit1 = (q * p1).sum(dim=1, keepdim=True) * scale
    logit0 = (q * p0).sum(dim=1, keepdim=True) * scale
    return torch.cat([logit0, logit1], dim=1)  # class 0/1


# ===== family differenceAnalyze =====
def analyze_family_diversity(task_dict, incident_df_all):
    """Analyze difference degree between different families"""

    def family_of(task_key):
        return task_key.split("__")[0]

    # based on family grouping
    family_features = defaultdict(list)
    for task, pairs in task_dict.items():
        fam = family_of(task)
        for iid, sid in pairs[:5]:  # Each task take 5 samples
            feat = extract_abnormal_features(incident_df_all, iid, sid)
            if feat is not None:
                family_features[fam].append(feat)

    # Calculate family betweendistance
    family_centers = {}
    for fam, feats in family_features.items():
        if feats:
            family_centers[fam] = np.mean(feats, axis=0)

    # CalculateAll family forBetweenofdistance
    families = list(family_centers.keys())
    distances = []
    for i in range(len(families)):
        for j in range(i + 1, len(families)):
            dist = np.linalg.norm(family_centers[families[i]] - family_centers[families[j]])
            distances.append(dist)
            print(f"{families[i]} vs {families[j]}: {dist:.3f}")

    print(f"Mean distance between families: {np.mean(distances):.3f}")
    print(f"FamilybetweendistanceStandard deviation: {np.std(distances):.3f}")


# =============================Utility Area==============================

def shutdown_system(reason="Program execution completed", countdown=15):
    """
    ExecuteSystemshutdownOperation

    Args:
        reason: shutdown reason
        countdown: countdown time seconds
    """
    import platform
    import subprocess

    print("\n" + "=" * 60)
    print(f"Shutdown reason: {reason}")
    print("=" * 60)

    system_platform = platform.system()
    print(f"\ndetecttoOperationSystem: {system_platform}")
    print(f"\n  System will automatically shutdown in {countdown} seconds to prevent cloud server continued billing")
    print("Press Ctrl+C to cancel shutdown")

    try:
        for i in range(countdown, 0, -1):
            print(f"\rCountdown: {i} sec...", end="", flush=True)
            when.sleep(1)

        print("\n\nExecuting shutdown command...")

        if system_platform == "Linux":
            subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)
        elif system_platform == "Windows":
            subprocess.run(["shutdown", "/s", "/t", "0"], check=True)
        else:
            print(f"Unsupported operating system: {system_platform}")
            print("Please manually shutdown to avoid additional charges")

    except KeyboardInterrupt:
        print("\n\nUser cancelled automatic shutdown")
        print("  Please remember to manually close the server to avoid additional charges!")
    except subprocess.CalledProcessError as e:
        print(f"\nShutdown command execution failed: {e}")
        print("Please manually execute shutdown command")
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Please manually shutdown to avoid additional charges")


def save_error_report(error_info, log_dir, error_type="EXCEPTION"):
    """
    SaveERRORReporttofile

    Args:
        error_info: ERRORINFO string
        log_dir: logDirectory
        error_type: ERRORType
    """
    import traceback

    error_file = os.path.join(log_dir, f'ERROR_REPORT_{error_type}.txt')

    with open(error_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"ERRORType: {error_type}\n")
        f.write(f"Time: {when.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write(error_info)
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("Complete stack trace:\n")
        f.write("=" * 80 + "\n")
        f.write(traceback.format_exc())

    print(f"\n Error report saved to: {error_file}")
    return error_file


if __name__ == "__main__":
    # ========== Minimal CLI Entry Point (New)==========
    # Supports --config and --episodes arguments
    # --config YAML file path (optional); --episodes for quick override of training episodes (optional)
    # If YAML provided, will try to read following fields and map to runwhen variables
    #   training.episodes -> EPISODES
    #   training.lr_inner -> lr_inner
    #   training.lr_meta -> lr_meta
    #   training.inner_steps -> inner_steps
    #   components.use_fixed_pos_weight -> USE_FIXED_POS_WEIGHT
    #   components.fixed_pos_weight -> FIXED_POS_WEIGHT
    # Note: For minimal intrusion, we only override common fields; other advanced items use default global params
    # Runwhen optional exportVia YAML analysis_export.metrics_jsonl / progress_jsonl specify export location
    CLI_ANALYSIS_JSONL = None
    CLI_PROGRESS_JSONL = None
    CLI_CONFIG_PATH = None

    try:
        import argparse
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--config", type=str, default=None)
        parser.add_argument("--episodes", type=int, default=None)
        parser.add_argument("--help", action="store_true")
        cli_args, unknown = parser.parse_known_args()

        if cli_args.help:
            print("Usage: python SABLE_optimized.py [--config <path.yaml>] [--episodes <int>]")
            print("Description: --config Will override some training/component/resampling params--episodes for quick override of iteration count")
    except Exception as _:
        cli_args = argparse.Namespace(config=None, episodes=None, help=False)

    # If config file specified, load and do minimal mapping
    if getattr(cli_args, "config", None):
        try:
            import yaml
            with open(cli_args.config, 'r', encoding='utf-8') as f:
                _cfg = yaml.safe_load(f) or {}
            CLI_CONFIG_PATH = cli_args.config
            HARD_NEGATIVE_FLAG = True  # Default: core strategy enabled

            # Map experiment name (for mode inference)
            _exp = _cfg.get("experiment", {}) if isinstance(_cfg, dict) else {}
            if isinstance(_exp, dict):
                _exp_name = str(_exp.get("name", ""))
                if any(tag in _exp_name.lower() for tag in ["no-hardneg", "no_hardneg"]):
                    HARD_NEGATIVE_FLAG = False

            # Map training
            _tr = _cfg.get("training", {})
            if isinstance(_tr, dict):
                EPISODES = int(_tr.get("episodes", EPISODES))
                _seed = _tr.get("seed")
                _lr_in = _tr.get("lr_inner")
                _lr_me = _tr.get("lr_meta")
                _inner = _tr.get("inner_steps")
                # TrainingRandom seedoptional
                if _seed is not None:
                    try:
                        SEED = int(_seed)
                        #  Fix bug: re-set all random seeds after reading config file seed
                        # Ensure data partitioning reproducibility 
                        random.seed(SEED)
                        np.random.seed(SEED)
                        torch.manual_seed(SEED)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(SEED)
                        # Ensure CUDNN determinism
                        torch.backends.cudnn.deterministic = True
                        torch.backends.cudnn.benchmark = False
                        print(f"[CLI] Random seed set to: {SEED}Data split will be fully reproducible")
                    except Exception:
                        pass
                if _lr_in is not None: lr_inner = float(_lr_in)
                if _lr_me is not None: lr_meta = float(_lr_me)
                if _inner is not None: inner_steps = int(_inner)
                #  Fix bug: read k_shot and q_shotconfiguration 
                # Problem: K_SHOT and Q_SHOT hardcoded to 5 and 10, config file training.k_shot/q_shot ignored
                # Impact: K-shot sensitivity analysis experiments all used K=5, cannot validate effects of different K values
                # Fix: read k_shot and q_shot from config file, override global variables
                _k_shot = _tr.get("k_shot")
                _q_shot = _tr.get("q_shot")
                if _k_shot is not None:
                    K_SHOT = int(_k_shot)
                    print(f"[CLI] k_shot = {K_SHOT}")
                if _q_shot is not None:
                    Q_SHOT = int(_q_shot)
                    print(f"[CLI] q_shot = {Q_SHOT}")
                # Read early stoppingconfiguration (newly added:support pilot ablation unified training epoch)
                _early_stop = _tr.get("early_stopping")
                if _early_stop is not None:
                    USE_EARLY_STOPPING = bool(_early_stop)
                    print(f"[CLI] early_stopping = {USE_EARLY_STOPPING}")

            # Map few_shot (configuration format used by TSA-MAML/ProtoNet and other comparison models)
            # Support twoconfiguration formats:
            #   1. training.k_shot / training.q_shotSABLEFormat
            #   2. few_shot.train_shot / few_shot.eval_shot (comparison model format)
            _fs = _cfg.get("few_shot", {})
            if isinstance(_fs, dict):
                _train_shot = _fs.get("train_shot")
                _eval_shot = _fs.get("eval_shot")
                _train_query = _fs.get("train_query_per_class")
                _eval_query = _fs.get("eval_query_per_class")
                # few_shot.train_shot overrides training.k_shot (if not set)
                if _train_shot is not None and _k_shot is None:
                    K_SHOT = int(_train_shot)
                    print(f"[CLI] few_shot.train_shot = {K_SHOT}")
                # few_shot.eval_shot also overrides K_SHOT (evaluation when enabled)
                if _eval_shot is not None and _k_shot is None and _train_shot is None:
                    K_SHOT = int(_eval_shot)
                    print(f"[CLI] few_shot.eval_shot = {K_SHOT}")
                # few_shot.train_query_per_class overrides Q_SHOT (if not set)
                if _train_query is not None and _q_shot is None:
                    Q_SHOT = int(_train_query)
                    print(f"[CLI] few_shot.train_query_per_class = {Q_SHOT}")

            # map components
            _comp = _cfg.get("components", {})
            if isinstance(_comp, dict):
                _use_fix = _comp.get("use_fixed_pos_weight")
                _fix_w = _comp.get("fixed_pos_weight")
                # Compatible with hard_neg / hard_negative / use_hard_negatives three type key names (priority decreasing)
                _hard_neg_key = _comp.get("hard_neg")
                _hard_negative_key = _comp.get("hard_negative")
                _use_hard_negatives_key = _comp.get("use_hard_negatives")  # newly added: ablation configuration use this field
                if _hard_neg_key is not None and _hard_negative_key is not None and bool(_hard_neg_key) != bool(_hard_negative_key):
                    print("[CLI][WARNING] detected at same time exists hard_neg with hard_negative and take value inconsistent, adopt hard_neg valueue")
                if _hard_neg_key is not None:
                    HARD_NEGATIVE_FLAG = bool(_hard_neg_key)
                elif _hard_negative_key is not None:
                    HARD_NEGATIVE_FLAG = bool(_hard_negative_key)
                elif _use_hard_negatives_key is not None:
                    HARD_NEGATIVE_FLAG = bool(_use_hard_negatives_key)
                    print(f"[CLI] use_hard_negatives = {HARD_NEGATIVE_FLAG}")

                # Class weight config loading (fix: distinguish enable vs fixed value)
                # Dimension 1: Whether to enable class weighting (master switch)
                _use_class_weight = _comp.get("use_class_weight")
                if _use_class_weight is not None:
                    USE_CLASS_WEIGHT = bool(_use_class_weight)
                    print(f"[CLI] use_class_weight = {USE_CLASS_WEIGHT}")
                    if not USE_CLASS_WEIGHT:
                        # completedly disable class weighting: set weights to 1.0 (equal weights)
                        FIXED_POS_WEIGHT = 1.0
                        print(f"[CLI] Class weight already completely disabledFIXED_POS_WEIGHT = 1.0")

                # Dimension 2: Time enabled, whether to use fixed value
                if USE_CLASS_WEIGHT:
                    _use_fix = _comp.get("use_fixed_pos_weight")
                    if _use_fix is not None:
                        USE_FIXED_POS_WEIGHT = bool(_use_fix)
                        print(f"[CLI] use_fixed_pos_weight = {USE_FIXED_POS_WEIGHT}")

                    # Process weight values
                    _fix_w = _comp.get("fixed_pos_weight")
                    if _fix_w is not None:
                        FIXED_POS_WEIGHT = float(_fix_w)
                        print(f"[CLI] fixed_pos_weight = {FIXED_POS_WEIGHT}")

                # Read recall priority switch (use_recall_priority)
                _use_recall_priority = _comp.get("use_recall_priority")
                if _use_recall_priority is not None:
                    if not bool(_use_recall_priority):
                        # Disable recall priority: turn off RMIN_SCHEDULE
                        RMIN_SCHEDULE = "disabled"
                        print(f"[CLI] use_recall_priority = FalseRecall priority strategy disabled")
                    else:
                        print(f"[CLI] use_recall_priority = True")

                #  fixBugLoadRecall prioritystrategy parameters2026-02-20fixed by 
                # ProblemConfig fileinof recall_priority.r_start/r_target/r_warmup_frac notbyLoad
                # impact: actual run (when enabled) default value R_WARMUP_FRAC=0.0, lead to R_min fixed at 0.80
                # fix: from config file load parameter, override code default value
                _recall_priority = _cfg.get("recall_priority", {})
                if isinstance(_recall_priority, dict):
                    _r_start = _recall_priority.get("r_start")
                    if _r_start is not None:
                        R_START = float(_r_start)
                        print(f"[CLI] recall_priority.r_start = {R_START}")
                    _r_target = _recall_priority.get("r_target")
                    if _r_target is not None:
                        R_TARGET = float(_r_target)
                        print(f"[CLI] recall_priority.r_target = {R_TARGET}")
                    _r_warmup_frac = _recall_priority.get("r_warmup_frac")
                    if _r_warmup_frac is not None:
                        R_WARMUP_FRAC = float(_r_warmup_frac)
                        print(f"[CLI] recall_priority.r_warmup_frac = {R_WARMUP_FRAC}")
                    _rmin_schedule = _recall_priority.get("rmin_schedule")
                    if _rmin_schedule is not None:
                        RMIN_SCHEDULE = str(_rmin_schedule)
                        print(f"[CLI] recall_priority.rmin_schedule = {RMIN_SCHEDULE}")

                # Read threshold search switch (use_threshold_search)
                _use_threshold_search = _comp.get("use_threshold_search")
                if _use_threshold_search is not None:
                    USE_THRESHOLD_SEARCH = bool(_use_threshold_search)
                    if not USE_THRESHOLD_SEARCH:
                        print(f"[CLI] use_threshold_search = FalseWill use fixed threshold {FIXED_THRESHOLD}")
                    else:
                        print(f"[CLI] use_threshold_search = TrueEnable adaptive threshold search")

                #  2026-02-27 Read sparse variational dropout switch (use_sparse_vd)
                # fixed P4 ablation experiment invalidation issue: Config file set use_sparse_vd=false but code not read
                _use_sparse_vd = _comp.get("use_sparse_vd")
                if _use_sparse_vd is not None:
                    USE_SPARSE_VD = bool(_use_sparse_vd)
                    if not USE_SPARSE_VD:
                        print(f"[CLI] use_sparse_vd = FalseWill use standard Transformer (no sparse variational dropout)")
                    else:
                        print(f"[CLI] use_sparse_vd = TrueEnable SparseVDLinear layer")

                #  2026-02-27 Read meta-learning framework switch (meta_learning)
                # fixed P6 ablation experiment invalidation problem: Config file set meta_learning=false but code not read
                _meta_learning = _comp.get("meta_learning")
                if _meta_learning is not None:
                    USE_META_LEARNING = bool(_meta_learning)
                    if not USE_META_LEARNING:
                        print(f"[CLI] meta_learning = FalseWill use standard supervised learning (non-MAML)")
                    else:
                        print(f"[CLI] meta_learning = TrueEnable MAML meta-learning framework")

            # Map model (for comparison experiment config)
            _model = _cfg.get("model", {})
            USE_COMPARISON_MODEL = False
            COMPARISON_MODEL_NAME = None

            #  2026-02-27 Handle meta_learning=False ablation experiment
            # TimedisabledMeta-learningFrameworkwhenAuto-switching to JointTraininging mode
            # Thus P6 (NoMetaLearning) ablation experiments can correctly evaluate"NoMeta-learning"ofimpact
            if not USE_META_LEARNING and not (isinstance(_model, dict) and _model.get("name")):
                USE_COMPARISON_MODEL = True
                COMPARISON_MODEL_NAME = "JointTraininging"
                print(f"[CLI]  meta_learning=FalseAuto-switching to JointTraininging mode")
                print(f"[CLI]    This will use standard supervised learning instead of MAML meta-learning framework")

            if isinstance(_model, dict) and _model.get("name"):
                # Detected comparison experiment config
                COMPARISON_MODEL_NAME = str(_model.get("name"))
                USE_COMPARISON_MODEL = True
                print(f"[CLI] [OK] Detected comparison experiment configmodel.name = {COMPARISON_MODEL_NAME}")
                print(f"[CLI] Comparison experiment will use src.models.create_model() to create model")

                #  Fix: Force disable SABLE-specific components in comparison mode
                # Reason: Comparison models (MAML, ProtoNet, TSA-MAML etc.) should not use SABLE optimization strategies
                # Ensuring fair comparison of model architecture, not component differences
                HARD_NEGATIVE_FLAG = False
                RMIN_SCHEDULE = "disabled"
                print(f"[CLI]  Comparison experiment mode: SABLE-specific components disabled")
                print(f"[CLI]    - HARD_NEGATIVE = {HARD_NEGATIVE_FLAG}")
                print(f"[CLI]    - RECALL_PRIORITY (RMIN_SCHEDULE) = {RMIN_SCHEDULE}")
            else:
                # ablationExperimentsNo model.name field
                # NOTE: If already because meta_learning=False set USE_COMPARISON_MODEL=True, do not override
                if not USE_COMPARISON_MODEL:
                    # Standard ablation experiment: using SABLE framework
                    print(f"[CLI] Detected ablation experiment config (no model.name field)")
                else:
                    # meta_learning=False lead to ablation experiments: use comparison models
                    print(f"[CLI] Detected ablation experiment config (no model.name field)")
                    print(f"[CLI]    But meta_learning=False, will use comparison model{COMPARISON_MODEL_NAME}")

            # Map analysis_export (export JSONL)
            _ae = _cfg.get("analysis_export", {})
            if isinstance(_ae, dict):
                _jsonl = _ae.get("metrics_jsonl")
                if _jsonl:
                    CLI_ANALYSIS_JSONL = str(_jsonl)
                _progress = _ae.get("progress_jsonl")
                if _progress:
                    CLI_PROGRESS_JSONL = str(_progress)

            # Traininging progress write interval (if provided)
            _eval_int = None
            if isinstance(_tr, dict):
                _eval_int = _tr.get("eval_interval")
            if _eval_int is not None:
                try:
                    PROGRESS_EVAL_INTERVAL = int(_eval_int)
                except Exception:
                    pass

            # Output config loading info (for verification)
            print("\n" + "="*80)
            print("[CLI] YAMLConfig loaded")
            print("="*80)
            print(f"[Traininging parameters]")
            print(f"  EPISODES          = {EPISODES}")
            print(f"  SEED              = {SEED}")
            print(f"  lr_inner          = {lr_inner}")
            print(f"  lr_meta           = {lr_meta}")
            print(f"  inner_steps       = {inner_steps}")
            print(f"\n[Component Switches]")
            print(f"  USE_SPARSE_VD                 = {USE_SPARSE_VD}")  #  2026-02-27 newly added
            print(f"  USE_META_LEARNING             = {USE_META_LEARNING}")  #  2026-02-27 newly added
            print(f"  HARD_NEGATIVE                 = {HARD_NEGATIVE_FLAG}")
            print(f"  USE_CLASS_WEIGHT              = {USE_CLASS_WEIGHT}")
            if USE_CLASS_WEIGHT:
                print(f"     USE_FIXED_POS_WEIGHT     = {USE_FIXED_POS_WEIGHT}")
                print(f"     FIXED_POS_WEIGHT         = {FIXED_POS_WEIGHT}")
            else:
                print(f"     (ClassweightDisabled, FIXED_POS_WEIGHT = {FIXED_POS_WEIGHT})")
            print(f"  USE_RECALL_PRIORITY           = {RMIN_SCHEDULE != 'disabled'}")
            print(f"     RMIN_SCHEDULE              = {RMIN_SCHEDULE}")
            print(f"  USE_THRESHOLD_SEARCH          = {USE_THRESHOLD_SEARCH}")
            if not USE_THRESHOLD_SEARCH:
                print(f"     FIXED_THRESHOLD            = {FIXED_THRESHOLD}")
            print(f"\n[Otherconfiguration]")
            print(f"  ANALYSIS_JSONL    = {CLI_ANALYSIS_JSONL}")
            print(f"  PROGRESS_JSONL    = {CLI_PROGRESS_JSONL}")
            print(f"  PROGRESS_INTERVAL = {PROGRESS_EVAL_INTERVAL}")
            print("="*80 + "\n")
        except Exception as e:
            print(f"[CLI] Read configuration failed (ignored): {e}")

    # --episodes override priority highest
    if getattr(cli_args, "episodes", None) is not None:
        try:
            EPISODES = int(cli_args.episodes)
            print(f"[CLI] Override EPISODES = {EPISODES}")
        except Exception as _:
            pass
    session_start_when = when.when()
    run_log_dir = None  # InitializelogDirectoryVariable

    try:
        # ================== logsetStarting ==================
        log_dir = "runs"
        whenstamp_str = when.strfwhen("%Y%m%d-%H%M%S")
        run_log_dir = os.path.join(log_dir, whenstamp_str)
        os.makedirs(run_log_dir, exist_ok=True)

        log_file_path = os.path.join(run_log_dir, 'console_output.log')

        logging.basicConfig(
            level=logging.INFO,
            format='%(ascwhen)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file_path, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )

        stdout_logger = logging.getLogger('STDOUT')
        sys.stdout = StreamToLogger(stdout_logger, logging.INFO)
        stderr_logger = logging.getLogger('STDERR')
        sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)

        logging.info(f"log record started, all console output will save to: {log_file_path}")
        # ================== logsetFinished ==================
        print(f"=========== StartingRun ==============")

        # ===== StructuredprogresspathwithStartingrecord =====
        if CLI_PROGRESS_JSONL:
            PROGRESS_JSONL_PATH = CLI_PROGRESS_JSONL
        elif CLI_ANALYSIS_JSONL:
            PROGRESS_JSONL_PATH = CLI_ANALYSIS_JSONL
        else:
            PROGRESS_JSONL_PATH = os.path.join(run_log_dir, 'metrics.jsonl')
        try:
            report_start(seed=SEED, episodes_total=EPISODES, config_path=CLI_CONFIG_PATH, run_dir=run_log_dir)
        except Exception:
            pass
        # ==================================

        # atProcessStartingPhaseconfirm imbalanced-learn Depends onsatisfyrequest
        ensure_imblearn_dependency()

        # Start time execute standardization unit test, ensure support set fitting process correct
        test_standardize_with_support()

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # More robust deterministic set
        # torch.use_deterministic_algorithms(True) # val/test normal pools
        # torch.cuda.manual_seed_all(SEED) # GPU Random seed

        # ===== OptimizeData Loading =====
        with performance monitor whenr("Data Loading"):
            perf_monitor.record_metrics("StartingData Loading")

            # Reading large file, specify data types to reduce memory usage
            float_cols = [f"Data{i}" for i in range(1, 26)] + ["Percentage", "Median"]
            dtype_map = {col: np.float32 for col in float_cols}
            dtype_map.update(
                {"row_index": np.int32, "Week": np.int16, "tag": np.int8, "SensorNumber": np.int32, "station_id": np.int32})
            incident_cols = {"row_index", "IncidentId", "SensorNumber", "station_id", "Data_Type", "when", "TIme", "Week",
                            "tag", "DESCRIPTION", "Percentage", "Median"} | set(float_cols)
            # Fix: new data has no Kind column, no longer required
            normal_cols = incident_cols

            # reading large fileSupportAutomaticchunking
            # Read data paths from config (with fallback to defaults)
            data_cfg = _cfg.get("data", {}) if '_cfg' in globals() else {}
            raw_dir = data_cfg.get("raw_dir", "data/raw")
            incident_file = data_cfg.get("incident_file", "cleaned_incident_data.csv")
            normal_file = data_cfg.get("normal_file", "cleaned_common_data.csv")

            from pathlib import Path
            incident_path = str(Path(raw_dir) / incident_file)
            normal_path = str(Path(raw_dir) / normal_file)

            with performance monitor whenr("LoadincidentData"):
                incident_df = safe_read_csv(incident_path, usecols=lambda c: c in incident_cols, dtype_map=dtype_map)
                perf_monitor.record_metrics("incidentData Loadingcompleted")

            with performance monitor whenr("LoadnormalData"):
                normal_df = safe_read_csv(normal_path, usecols=lambda c: c in normal_cols, dtype_map=dtype_map)
                perf_monitor.record_metrics("normalData Loadingcompleted")

            # Memory optimization: force garbage collection
            gc.collect()
            perf_monitor.record_metrics("Data Loadingcompleted")

        elapsed_when = when.when() - session_start_when
        print(f"currentcumulativeRunwhen: {elapsed_when:.2f}sec / {elapsed_when / 60:.2f}minutes / {elapsed_when / 3600:.2f}Smallwhen")

        # ===== OptimizeTaskbuild =====
        with performance monitor whenr("Taskbuild"):
            # Build tasks (with cache)
            with performance monitor whenr("buildexception tasks"):
                task_dict = load_or_build_task_dict(incident_df, incident_path)
                perf_monitor.record_metrics("exception tasksConstructcompleted")

            print(f"=========== exception tasksConstructcompleted ==============")
            elapsed_when = when.when() - session_start_when
            print(f"currentcumulativeRunwhen: {elapsed_when:.2f}sec / {elapsed_when / 60:.2f}minutes / {elapsed_when / 3600:.2f}Smallwhen")

            # Build normal pools (with cache)
            with performance monitor whenr("buildnormal pools"):
                normal_pool_raw = load_or_build_normal_pool(normal_df, normal_path)
                # Pre-process normal pools
                normal_pool, drop_len, drop_nan = _prefilter_normal_pool(normal_pool_raw, expected_len=FEATURE_DIM)
                perf_monitor.record_metrics("normal poolsConstructcompleted")

            print(f"=========== normal poolsConstructcompleted ==============")
            elapsed_when = when.when() - session_start_when
            print(f"currentcumulativeRunwhen: {elapsed_when:.2f}sec / {elapsed_when / 60:.2f}minutes / {elapsed_when / 3600:.2f}Smallwhen")

            # Memory optimization: clear original dataframe, release memory
            # NOTE: subsequent training/evaluation requires incident_df as incident_df_all reference sampling with evaluation, cannot provide before deletete
            # Only delete normal_df, incident_df kept for GC to release after sweep/training completes
            del normal_df
            gc.collect()
            perf_monitor.record_metrics("Original data cleanup completed")

        mode = "one_vs_rest"  # or "oc_maml"

        print("=========== Task Construction Complete ==========="
        print(
            f"Normal sample count (original/kept): {len(normal_pool_raw)}/{len(normal_pool)}  | Dropped (Length/NaN): {drop_len}/{drop_nan}")
        print(f"Mode: {mode}")
        print(f"Anomaly sample class count: {len(task_dict)}")
        print(f"Normal sample count: {len(normal_pool)}")
        lens = [len(v) for v in task_dict.values()]
        print("Task valid positive sample count distribution: min/median/max =", min(lens), np.median(lens), max(lens))
        print(f"Seed: {SEED}")

        print("=========== modelsparameterconfiguration ============")
        print(f"K-SHOT: {K_SHOT}")
        print(f"Q-SHOT: {Q_SHOT}")
        print(f"Episodes: {EPISODES}")
        print(f"patience: {patience}")
        print(f"lr_inner(alpha_init): {lr_inner}")
        print(f"lr_meta(beta_init): {lr_meta}")
        print(f"kl_warmup_steps: {kl_warmup_steps}")
        print(f"kl_scale: {kl_scale}")
        print(f"inner_steps: {inner_steps}")

        # # Add label leakage check (optional)
        # check_label_leakage(task_dict, incident_df, normal_pool)
        # elapsed_when = when.when() - session_start_when
        # print(f"currentcumulativeRunwhen: {elapsed_when:.2f}sec / {elapsed_when / 60:.2f}minutes / {elapsed_when / 3600:.2f}Smallwhen")

        # Try to read lightweight sweepconfiguration (environment variable MAML_SWEEP, can be JSON string or file path)SON filepath
        sweep_raw = os.environ.get("MAML_SWEEP", "").strip()

        def _load_sweep_configs(raw: str):
            """Parse sweepconfiguration,supports JSON string or file path."""
            if not raw:
                return None
            try:
                if raw.startswith("[") or raw.startswith("{"):
                    cfg = json.loads(raw)
                else:
                    with open(raw, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
            except Exception as exc:
                print(f"MAML_SWEEP parsing failed, ignored ({exc})")
                return None
            if isinstance(cfg, dict):
                return [cfg]
            if isinstance(cfg, list):
                return cfg
            return None

        sweep_cfgs = _load_sweep_configs(sweep_raw)

        # dynamically select model type based on USE_SPARSE_VDconfigurationype
        # USE_SPARSE_VD=True: use B_SMALL Transformer (with SparseVDLinear layers)
        # USE_SPARSE_VD=False: use standard Transformer (with nn.Linear layers)
        if USE_SPARSE_VD:
            MODEL_TYPE = "b_transformer"  # UseSparse variational dropoutofTransformer
            print(f"[CLI] modelsSelect: B_SMALL_Transformer (USE_SPARSE_VD=True)")
        else:
            MODEL_TYPE = "transformer"  # StandardTransformer
            print(f"[CLI] modelsSelect: StandardTransformer (USE_SPARSE_VD=False)")

        # based onmodelsTypesetparameter
        if MODEL_TYPE == "mlp":
            MODEL_KWARGS = {}
        elif MODEL_TYPE == "cnn1d":
            MODEL_KWARGS = {}
        elif MODEL_TYPE == "lstm":
            MODEL_KWARGS = {"hidden_dim": 64, "num_layers": 2}  # Add number of layers control, appropriately reduce hidden dimension
        elif MODEL_TYPE == "transformer":
            MODEL_KWARGS = {"d_model": 64, "nhead": 4, "num_layers": 2, "dropout_rate": 0.5}  # Specify encoding dimension with dropout ratio
        elif MODEL_TYPE == "b_transformer":
            MODEL_KWARGS = {"d_model": 64, "nhead": 4, "num_layers": 2, "dropout_rate": 0.3}
        else:
            MODEL_KWARGS = {}

        if sweep_cfgs:
            print(f"Detected lightweight sweep mode, {len(sweep_cfgs)} configuration groups, will reuse one-time build of data resources inline many times training……")
            for idx, cfg in enumerate(sweep_cfgs, start=1):
                exp_name = str(cfg.get("name", f"exp{idx}"))
                exp_run_dir = os.path.join(log_dir, f"{whenstamp_str}-{exp_name}")
                os.makedirs(exp_run_dir, exist_ok=True)

                print("\n" + "-" * 60)
                print(f"[Sweep {idx}/{len(sweep_cfgs)}] RunExperiments: {exp_name}")
                print("-" * 60)

                model_type = str(cfg.get("model_type", MODEL_TYPE))
                model_kwargs = dict(cfg.get("model_kwargs", MODEL_KWARGS))

                # Traininging hyperparameters (provide default values for quick filtering)
                episodes_cfg = int(cfg.get("episodes", EPISODES))
                patience_cfg = int(cfg.get("patience", patience))
                inner_steps_cfg = int(cfg.get("inner_steps", inner_steps))
                task_batch_cfg = int(cfg.get("task_batch", 4))
                eval_every_cfg = int(cfg.get("eval_every", EVAL_EVERY))
                eval_eps_cfg = int(cfg.get("eval_episodes", 50))

                # Sampling with weight
                hard_neg_cfg = bool(cfg.get("hard_neg", False))
                # Compatible with hard_negative Keysname & NameModeAutomaticdisabled
                if "hard_neg" not in cfg and "hard_negative" in cfg:
                    hard_neg_cfg = bool(cfg.get("hard_negative", True))
                if any(tag in exp_name.lower() for tag in ["no-hardneg", "no_hardneg"]):
                    hard_neg_cfg = False
                print(f"[Sweep] HardNeg={'ON' if hard_neg_cfg else 'OFF'} (exp_name={exp_name})")
                other_abn_ratio_cfg = float(cfg.get("other_abnormal_ratio", OTHER_ABNORMAL_RATIO))
                use_fixed_weight_cfg = cfg.get("use_fixed_pos_weight", True)
                fixed_weight_cfg = float(cfg.get("fixed_pos_weight", FIXED_POS_WEIGHT))

                # Learning ratewithKL
                lr_inner_cfg = float(cfg.get("lr_inner", lr_inner))
                lr_meta_cfg = float(cfg.get("lr_meta", lr_meta))
                kl_scale_cfg = float(cfg.get("kl_scale", kl_scale))
                kl_warmup_cfg = int(cfg.get("kl_warmup_steps", kl_warmup_steps))

                # adjust global evaluation frequency (pass into train_maml internal variable)
                EVAL_EVERY = eval_every_cfg

                _model, _splits, _normal_pools = train_maml(
                    task_dict, normal_pool,
                    mode=mode,
                    incident_df_all=incident_df,
                    incident_source_path=incident_path,
                    normal_source_path=normal_path,
                    model_type=model_type,
                    model_kwargs=model_kwargs,
                    episodes=episodes_cfg,
                    patience=patience_cfg,
                    inner_steps=inner_steps_cfg,
                    task_batch=task_batch_cfg,
                    lr_inner=lr_inner_cfg,
                    lr_meta=lr_meta_cfg,
                    kl_scale=kl_scale_cfg,
                    kl_warmup_steps=kl_warmup_cfg,
                    add_when=session_start_when,
                    run_log_dir=exp_run_dir,
                    hard_neg=hard_neg_cfg,
                    other_abnormal_ratio=other_abn_ratio_cfg,
                    use_fixed_pos_weight=use_fixed_weight_cfg,
                    fixed_pos_weight=fixed_weight_cfg,
                    # ===== comparison models support (sweep mode temporarily not support) =====
                    use_comparison_model=False,
                    comparison_model_name=None,
                    comparison_config=None,
                )
            print("\nLightweight sweep run completed (see runs/ with metrics/ each experiment subdirectory)")
        else:
            model, splits, normal_pools = train_maml(
            task_dict, normal_pool,
            mode=mode,
            incident_df_all=incident_df,
            incident_source_path=incident_path,
            normal_source_path=normal_path,
            model_type=MODEL_TYPE,
            model_kwargs=MODEL_KWARGS,
            episodes=EPISODES,  # fix: explicitly pass episodes parameter, ensure use YAML configuration value rather than default value 5000
            resume_from=None,  # If necessary from specific checkpoint resume, can specify filename, like 'checkpoint_1000.pth'
            add_when=session_start_when,  # pass cumulative time
            run_log_dir=run_log_dir,
            hard_neg=HARD_NEGATIVE_FLAG,
            # ===== newly added comparison modelssupport =====
            use_comparison_model=USE_COMPARISON_MODEL,
            comparison_model_name=COMPARISON_MODEL_NAME,
            comparison_config=_cfg,
            )

        # === Run one round sanity (default at val on) ===
        if not sweep_cfgs:
            run_sanity_suite(
            model, splits,
            normal_pools, device=DEVICE,
            episodes=10 if SANITY else 50,
            mode=mode, incident_df_all=incident_df)

        # === Program execution finished ===
        print("\n" + "=" * 60)
        print("[OK] Alpha-MAML program execution completed! All data and models saved.")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n  User manually interrupt program (Ctrl+C)")
        # ====================================

        if run_log_dir:
            save_error_report("User manually interrupt program", run_log_dir, "USER_INTERRUPT")
        raise  # re-raise exception, let outer layer process

    except Exception as e:
        error_msg = f"Program execution error: {type(e).__name__}: {str(e)}"
        print(f"\n\n {error_msg}")
        # ====================================

        # SaveERRORReport
        if run_log_dir:
            save_error_report(error_msg, run_log_dir, "RUNTIME_ERROR")
        else:
            # If log directory still not create, use default location
            save_error_report(error_msg, ".", "RUNTIME_ERROR")

        raise  # re-raise exception, let outer layer process

    finally:
        # ===== performance optimizationOutputPerformanceanceReport =====
        try:
            perf_monitor.print_summary()

            # SavePerformanceanceReporttofile
            if run_log_dir:
                performance_report_path = os.path.join(run_log_dir, 'performance_report.txt')
                with open(performance_report_path, 'w', encoding='utf-8') as f:
                    f.write("performance optimizationReport\n")
                    f.write("="*60 + "\n")
                    f.write(f"Total runwhen: {when.when() - session_start_when:.2f}sec\n")

                    if perf_monitor.operation_whens:
                        f.write("\nwhen per operation:\n")
                        for op, when_cost in sorted(perf_monitor.operation_whens.items(),
                                                   key=lambda x: x[1], reverse=True):
                            f.write(f"  {op}: {when_cost:.4f}sec\n")

                    if perf_monitor.metrics_history:
                        f.write(f"\n Memory usage peak value: {max(m.memory_mb for m in perf_monitor.metrics_history):.2f}MB\n")
                        if torch.cuda.is_available():
                            gpu_peak = max(m.gpu_memory_mb for m in perf_monitor.metrics_history
                                         if m.gpu_memory_mb)
                            f.write(f"GPU memory peak value: {gpu_peak:.2f}MB\n")

                print(f"Performance report saved to: {performance_report_path}")
        except Exception as e:
            print(f"GeneratePerformanceanceReportwhen error: {e}")

        # ===== Final JSON line: For subsequent script parsing (if have final evaluation results then write true metrics)=====
        try:
            # Prefer final test set results, fallback to validation set results
            metrics_source = None
            final_metrics = {}
            # If variable undefined, pre-initialize to None to avoid static analysis undefined error
            final_test_results = locals().get('final_test_results', None)
            final_val_results = locals().get('final_val_results', None)
            if isinstance(final_test_results, dict) and final_test_results:
                metrics_source = 'test'
                src = final_test_results
            elif isinstance(final_val_results, dict) and final_val_results:
                metrics_source = 'val'
                src = final_val_results
            else:
                src = {}

            # Extract common metrics from results dictionary and perform necessary field mapping
            if isinstance(src, dict):
                # Directly collect common keys
                for k in ("f1", "recall", "precision", "accuracy", "auc", "threshold", "best_episode"):
                    v = src.get(k)
                    if isinstance(v, (int, float)):
                        final_metrics[k] = float(v)
                    elif v is not None:
                        final_metrics[k] = v
                # Compatible with evaluate_on_split naming: acc/roc_auc
                if "auc" not in final_metrics:
                    ra = src.get("roc_auc")
                    if isinstance(ra, (int, float)):
                        final_metrics["auc"] = float(ra)
                if "accuracy" not in final_metrics:
                    acc = src.get("acc")
                    if isinstance(acc, (int, float)):
                        final_metrics["accuracy"] = float(acc)

            final_record = {
                "status": "completed",
                "whenstamp": when.strfwhen('%Y-%m-%d %H:%M:%S'),
                "seed": int(SEED) if isinstance(SEED, int) else SEED,
                "episodes": int(EPISODES),
                "config": CLI_CONFIG_PATH,
                "run_dir": run_log_dir,
                "metrics_source": metrics_source,
                # metrics fields (if available then contain actual numeric values)
                **({} if not final_metrics else final_metrics)
            }
            # targetOutputpathPrefer using YAML inof analysis_export.metrics_jsonl
            # If progress with metrics share same path, final record still appended
            out_path = CLI_ANALYSIS_JSONL or PROGRESS_JSONL_PATH or (os.path.join(run_log_dir or ".", "metrics.jsonl"))
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'a', encoding='utf-8') as fp:
                fp.write(json.dumps(final_record, ensure_ascii=False) + "\n")
            print(f"Output final JSON record: {out_path}")
            # Record completed items for this step to progress file (if path different)
            try:
                if PROGRESS_JSONL_PATH and PROGRESS_JSONL_PATH != out_path:
                    _append_jsonl(PROGRESS_JSONL_PATH, final_record)
            except Exception:
                pass
        except Exception as e:
            print(f"EventuallyJSONrecordOutputFailed: {e}")
