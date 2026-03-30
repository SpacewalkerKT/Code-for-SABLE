"""
BASE_LogReg: Logistic Regression Baseline

Reference: Logistic_Regression.md (Experiment reports/for comparison model research/)

Core Idea:
- Classical discriminative linear model for binary classification
- Freeze encoder parameters, only train logistic layer
- Joint training on all families (non-episodic)
- Zero-shot or simple support-set finetuning

Key Differences from Meta-Learning:
- No episode-based training (Joint Training approach)
- Frozen encoder (only logistic layer is trained)
- Optional: zero adaptation or simple support finetuning
- Direct binary classification on merged dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from .base import BaseComparisonModel
from . import register_model


@register_model("LogReg")
class LogRegModel(BaseComparisonModel):
    """
    Logistic Regression baseline for few-shot anomaly detection.

    Unified Constraints:
    - Encoder: Shared with SABLE (transformer_b_small), FROZEN
    - Threshold: Search mode, F1 metric
    - Resampling: Disabled
    - Task construction: N/A (non-episodic joint training)
    """

    def __init__(self, encoder: nn.Module, config: Dict[str, Any]):
        """
        Initialize Logistic Regression model.

        Args:
            encoder: Shared encoder (transformer_b_small)
            config: Model configuration from YAML
                - training.lr: Learning rate for logistic layer
                - training.weight_decay: L2 regularization (optional)
                - training.pos_weight: Positive class weight for BCE loss (optional)
                - training.adaptation_mode: "none" or "support_finetune"
                - training.adapt_steps: Steps for support finetuning (if enabled)
        """
        super().__init__(encoder, config)

        # Extract training hyperparameters
        training_config = config.get('training', {})

        self.lr = training_config.get('lr', 0.001)
        self.weight_decay = training_config.get('weight_decay', 1e-4)
        self.adaptation_mode = training_config.get('adaptation_mode', 'none')
        self.adapt_steps = training_config.get('adapt_steps', 5)
        # Whether to reset linear weights to global state after each task
        # True: per-task independent adaptation (like MAML/ProtoNet)
        # False: sequential adaptation across tasks (continual learning)
        self.reset_weights_per_task = training_config.get('reset_weights_per_task', True)

        # Validate hyperparameters
        if self.lr <= 0:
            raise ValueError(f"training.lr must be > 0, got {self.lr}")
        if self.weight_decay < 0:
            raise ValueError(f"training.weight_decay must be >= 0, got {self.weight_decay}")
        if self.adaptation_mode not in ['none', 'support_finetune']:
            raise ValueError(
                f"training.adaptation_mode must be 'none' or 'support_finetune', "
                f"got '{self.adaptation_mode}'"
            )
        if self.adapt_steps <= 0:
            raise ValueError(f"training.adapt_steps must be >= 1, got {self.adapt_steps}")
        if not isinstance(self.reset_weights_per_task, bool):
            raise TypeError(
                f"training.reset_weights_per_task must be bool, "
                f"got {type(self.reset_weights_per_task)}"
            )

        # Positive class weight for handling imbalanced data (optional)
        # Format: scalar weight for positive class
        pos_weight = training_config.get('pos_weight', None)
        if pos_weight is not None:
            if not isinstance(pos_weight, (int, float)):
                raise TypeError(
                    f"training.pos_weight must be numeric, "
                    f"got {type(pos_weight)}"
                )
            if pos_weight <= 0:
                raise ValueError(
                    f"training.pos_weight must be positive, "
                    f"got {pos_weight}"
                )
            # Use register_buffer for automatic device management
            self.register_buffer(
                'pos_weight',
                torch.tensor([pos_weight], dtype=torch.float32)
            )
        else:
            self.pos_weight = None

        # Freeze encoder parameters
        # This makes LogReg a "linear probe" on frozen features
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()  # Set to eval mode permanently

        # Build logistic layer: Linear(encoder_dim, 1)
        encoder_dim = self._get_encoder_dim()
        self.linear = nn.Linear(encoder_dim, 1)

        # Cache for global weights (used when reset_weights_per_task=True)
        self._global_linear_state = None

        # Optimizer for logistic layer only
        self.optimizer = torch.optim.Adam(
            self.linear.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        print(f" Initialized LogReg model:")
        print(f"  - Learning rate: {self.lr}")
        print(f"  - Weight decay: {self.weight_decay}")
        print(f"  - Positive class weight: {self.pos_weight}")
        print(f"  - Adaptation mode: {self.adaptation_mode}")
        if self.adaptation_mode == 'support_finetune':
            print(f"  - Adapt steps: {self.adapt_steps}")
            print(f"  - Reset weights per task: {self.reset_weights_per_task}")
        print(f"  - Encoder: FROZEN (linear probe mode)")

    def _get_encoder_dim(self) -> int:
        """Get encoder output dimension."""
        try:
            device = next(self.encoder.parameters()).device
            dummy_input = torch.randn(1, 32, device=device)

            with torch.no_grad():
                dummy_output = self.encoder(dummy_input)

            return dummy_output.shape[-1]
        except Exception as e:
            print(f"  Warning: Could not probe encoder output dim ({e}). "
                  f"Falling back to config default.")
            return self.config.get('model', {}).get('embedding_dim', 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through frozen encoder and logistic layer.

        Args:
            x: Input features [N, input_dim]

        Returns:
            logits: Binary classification logits [N]

        Note:
            Encoder parameters are frozen (requires_grad=False).
            Only the logistic layer is trainable.
        """
        with torch.no_grad():
            features = self.encoder(x)
        logits = self.linear(features).squeeze(-1)
        return logits

    def train_episode(self, support: Dict[str, torch.Tensor],
                      query: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Train on a single episode (joint training on merged data).

        Note: Similar to JointTraining, LogReg does NOT use episodic training.
        Instead, it treats support+query as a single batch for standard
        supervised learning.

        Args:
            support: Support set data (merged with query for training)
            query: Query set data (merged with support for training)

        Returns:
            metrics: Dictionary containing loss and accuracy

        Raises:
            ValueError: If both support and query are empty
        """
        self.linear.train()

        # Merge support and query into a single batch
        # This is standard Joint Training behavior
        x_all = torch.cat([support['x'], query['x']], dim=0)
        y_all = torch.cat([support['y'], query['y']], dim=0)

        # Guard against empty batch (prevent NaN or runtime errors)
        if len(y_all) == 0:
            raise ValueError(
                "Cannot train on empty batch: both support and query are empty. "
                "Check data filtering/sampling logic."
            )

        # Extract features (encoder is frozen, use no_grad for efficiency)
        with torch.no_grad():
            features = self.encoder(x_all)

        # Forward pass through logistic layer
        logits = self.linear(features).squeeze(-1)

        # Compute binary cross-entropy loss
        if self.pos_weight is not None:
            # Apply positive class weight
            loss = F.binary_cross_entropy_with_logits(
                logits, y_all.float(), pos_weight=self.pos_weight
            )
        else:
            loss = F.binary_cross_entropy_with_logits(logits, y_all.float())

        # Backward pass and optimization (only updates linear layer)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Compute metrics
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            # NOTE: 0.5 threshold is only for training accuracy metric
            # Test evaluation uses threshold search on validation set
            preds = (probs >= 0.5).long()
            labels = y_all.long()

            acc = (preds == labels).float().mean().item()

        return {
            'loss': loss.item(),
            'accuracy': acc,
            'batch_size': len(y_all)
        }

    def save_global_state(self) -> None:
        """
        Save current linear weights as global state.

        This is used when reset_weights_per_task=True to restore
        the globally trained weights after each task adaptation.
        """
        import copy
        self._global_linear_state = copy.deepcopy(self.linear.state_dict())

    def restore_global_state(self) -> None:
        """
        Restore linear weights from saved global state.

        Raises:
            RuntimeError: If global state was never saved
        """
        if self._global_linear_state is None:
            raise RuntimeError(
                "Cannot restore global state: global state was never saved. "
                "Call save_global_state() before first adapt()."
            )
        self.linear.load_state_dict(self._global_linear_state)

    def adapt(self, support: Dict[str, torch.Tensor]) -> None:
        """
        Adapt model to a new task.

        Two modes:
        1. "none" (default): Zero-shot, no adaptation (like JointTraining)
        2. "support_finetune": Simple finetuning on support set

        If reset_weights_per_task=True (default):
        - Before each adaptation, restores linear weights to global state
        - This ensures per-task independent adaptation (like MAML/ProtoNet)
        - External driver should call restore_global_state() after predict()

        If reset_weights_per_task=False:
        - Adapted weights persist across tasks (continual learning)

        Args:
            support: Support set for adaptation
                - 'x': Features [N_support, ...]
                - 'y': Labels [N_support]

        Raises:
            ValueError: If support set is empty in 'support_finetune' mode
        """
        # Guard against empty support set in finetuning mode
        if self.adaptation_mode == 'support_finetune' and len(support['y']) == 0:
            raise ValueError(
                "Cannot finetune on empty support set. "
                "Check task sampling/filtering logic or use adaptation_mode='none'."
            )

        if self.adaptation_mode == 'none':
            # Zero-shot baseline: no adaptation needed
            self.linear.eval()
            print(f" LogReg: No adaptation needed (zero-shot baseline)")
            print(f"  - Support size: {len(support['y'])} (unused)")
            return

        elif self.adaptation_mode == 'support_finetune':
            # Save global state on first adaptation
            if self._global_linear_state is None:
                self.save_global_state()

            # Restore from global state if per-task reset is enabled
            if self.reset_weights_per_task:
                self.restore_global_state()

            # Simple finetuning on support set
            self.linear.train()

            # Reset optimizer state to avoid interference from training phase
            # This ensures each task adaptation starts fresh without inherited momentum
            self.optimizer.state = {}

            x_sup = support['x']
            y_sup = support['y']

            for step in range(self.adapt_steps):
                # Extract features (encoder is frozen)
                with torch.no_grad():
                    features = self.encoder(x_sup)

                # Forward pass
                logits = self.linear(features).squeeze(-1)

                # Compute loss
                if self.pos_weight is not None:
                    loss = F.binary_cross_entropy_with_logits(
                        logits, y_sup.float(), pos_weight=self.pos_weight
                    )
                else:
                    loss = F.binary_cross_entropy_with_logits(logits, y_sup.float())

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.linear.eval()
            print(f" LogReg: Adapted to new task")
            print(f"  - Adaptation mode: support_finetune")
            print(f"  - Support size: {len(support['y'])}")
            print(f"  - Adapt steps: {self.adapt_steps}")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict anomaly scores for input samples.

        Uses the logistic layer (either globally trained or adapted).

        Args:
            x: Input features [N, ...]

        Returns:
            scores: Anomaly scores in [0, 1] [N]
        """
        self.linear.eval()

        with torch.no_grad():
            # Extract features (encoder is frozen)
            features = self.encoder(x)

            # Logistic forward: Linear → Sigmoid
            logits = self.linear(features).squeeze(-1)
            scores = torch.sigmoid(logits)

        return scores
