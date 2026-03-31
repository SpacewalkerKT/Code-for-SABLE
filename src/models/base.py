"""
Base comparison model interface for fair benchmarking against SABLE.

All comparison models (MAML, ProtoNet, TSA-MAML, JointTraining, LogReg)
must implement this interface to ensure:
- Unified encoder sharing
- Unified threshold search strategy
- Unified task construction
- Unified evaluation protocol
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn


class BaseComparisonModel(nn.Module, ABC):
    """
    Abstract base class for all comparison models.

    Unified Constraints (from execution manual):
    - All models share the same encoder structure (transformer_b_small)
    - All models use the same threshold search strategy (search mode, F1 metric)
    - All models disable training resampling (resampling.enabled=false)
    - Few-shot models share the same way/shot configuration
    """

    def __init__(self, encoder: nn.Module, config: Dict[str, Any]):
        """
        Initialize comparison model.

        Args:
            encoder: Shared encoder (transformer_b_small)
            config: Model configuration from YAML
        """
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.model_name = config.get('model', {}).get('name', 'Unknown')

    @abstractmethod
    def train_episode(self, support: Dict[str, torch.Tensor],
                      query: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Train on a single episode.

        Args:
            support: Support set data
                - 'x': Features [N_support, ...]
                - 'y': Labels [N_support]
            query: Query set data
                - 'x': Features [N_query, ...]
                - 'y': Labels [N_query]

        Returns:
            metrics: Dictionary containing loss and other metrics
        """
        pass

    @abstractmethod
    def adapt(self, support: Dict[str, torch.Tensor]) -> None:
        """
        Adapt model to a new task using support set.

        For meta-learning models (MAML, ProtoNet, TSA-MAML):
            - Perform task-specific adaptation (inner-loop update or prototype computation)
        For non-meta-learning models (JointTraining, LogReg):
            - No adaptation needed (zero-shot inference)

        Args:
            support: Support set for adaptation
                - 'x': Features [N_support, ...]
                - 'y': Labels [N_support]
        """
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict anomaly scores for input samples.

        Args:
            x: Input features [N, ...]

        Returns:
            scores: Anomaly scores in [0, 1] [N]
        """
        pass

    def evaluate(self, query: Dict[str, torch.Tensor],
                 threshold: Optional[float] = None) -> Dict[str, float]:
        """
        Evaluate model on query set.

        Args:
            query: Query set data
                - 'x': Features [N_query, ...]
                - 'y': True labels [N_query]
            threshold: Decision threshold (if None, use 0.5)

        Returns:
            metrics: Dictionary containing F1, recall, precision, etc.
        """
        self.eval()
        with torch.no_grad():
            scores = self.predict(query['x'])

            if threshold is None:
                threshold = 0.5

            preds = (scores >= threshold).long()
            labels = query['y'].long()

            # Calculate metrics
            tp = ((preds == 1) & (labels == 1)).sum().item()
            fp = ((preds == 1) & (labels == 0)).sum().item()
            fn = ((preds == 0) & (labels == 1)).sum().item()
            tn = ((preds == 0) & (labels == 0)).sum().item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        return {
            'f1': f1,
            'recall': recall,
            'precision': precision,
            'accuracy': accuracy,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'threshold': threshold
        }

    def search_threshold(self, val_data: Dict[str, torch.Tensor],
                         metric: str = 'f1',
                         grid: Optional[List[float]] = None) -> Tuple[float, Dict[str, float]]:
        """
        Search for optimal threshold on validation set.

        Unified threshold strategy (from execution manual):
        - Grid search over [0.1, 0.2, ..., 0.9]
        - Optimize for F1 score (or other metric)

        Args:
            val_data: Validation set data
                - 'x': Features
                - 'y': True labels
            metric: Metric to optimize ('f1', 'f2', etc.)
            grid: Threshold grid (if None, use default)

        Returns:
            best_threshold: Optimal threshold
            best_metrics: Metrics at optimal threshold
        """
        if grid is None:
            grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        best_threshold = 0.5
        best_score = -float('inf')
        best_metrics = None

        for threshold in grid:
            metrics = self.evaluate(val_data, threshold=threshold)
            score = metrics.get(metric, 0.0)

            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = metrics

        return best_threshold, best_metrics

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config

    def get_name(self) -> str:
        """Get model name."""
        return self.model_name

    # ===== SABLE training loop compatible interface =====
    # For using comparison models in SABLE.py training loop,
    # need to implement the following adaptation method.

    def named_params_dict(self):
        """
        Return parameter dictionary requiring gradients (compatible with SABLE training loop).

        Returns:
            OrderedDict: dictionary of parameter names -> parameter tensors
        """
        from collections import OrderedDict
        od = OrderedDict()
        for name, p in self.named_parameters():
            if p.requires_grad:
                od[name] = p
        return od

    def forward_with_weights(self, x: torch.Tensor, weights: dict) -> torch.Tensor:
        """
        Use specified weights for forward pass (compatible with SABLE MAML inner loop).

        Args:
            x: inputfeature [N, input_dim]
            weights: parameter dictionary (parameter name -> parameter tensor)

        Returns:
            logits: modeloutput [N]

        Note:
            This is an adaptation method, for using temporary weights in forward pass during MAML inner loop.
            For comparison models, we directly use self.forward(), because their parameters have through
            meta optimizer update, no need to manually replace weights.
        """
        # For comparison models, directly use current parameters for forward pass
        # Because their optimizer will automatically handle parameter updates
        return self.forward(x)

    def get_regularization_loss(self) -> torch.Tensor:
        """
        Calculate regularization loss (compatible with SABLE training loop).

        Returns:
            loss: regularization loss (scalar tensor)

        Note:
            For comparison models, we return 0, because:
            1. Comparison models use their own regularization method (such as L2 regularization)
            2. No need for additional KL divergence or Bayesian regularization
            3. Keep simple, avoid conflict with SABLE's complex regularization logic
        """
        # Return 0, indicating comparison models do not need additional regularization
        device = next(self.parameters()).device
        return torch.tensor(0.0, device=device)
