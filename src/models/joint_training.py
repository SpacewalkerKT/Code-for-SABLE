"""
BASE_JointTraining: Global Binary Classifier Baseline

Reference: Joint_Training.md (Experiment reports/for comparison model research/)

Core Idea:
- Non-meta-learning baseline (no episodes, no adaptation)
- Train a global binary classifier on ALL training families
- Merge all anomaly samples from all families into one positive class
- Test by zero-shot inference (no task-specific adaptation)

Key Differences from Meta-Learning:
- No episode-based sampling
- No inner-loop/outer-loop
- No support/query split
- Direct binary classification on merged dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from .base import BaseComparisonModel
from . import register_model


@register_model("JointTraining")
class JointTrainingModel(BaseComparisonModel):
    """
    Joint Training baseline for few-shot anomaly detection.

    Unified Constraints:
    - Encoder: Shared with SABLE (transformer_b_small)
    - Threshold: Search mode, F1 metric
    - Resampling: Disabled
    - Task construction: N/A (non-episodic)
    """

    def __init__(self, encoder: nn.Module, config: Dict[str, Any]):
        """
        Initialize Joint Training model.

        Args:
            encoder: Shared encoder (transformer_b_small)
            config: Model configuration from YAML
                - training.lr: Learning rate for global optimization
                - training.class_weights: Optional class weights for BCE loss
        """
        super().__init__(encoder, config)

        # Extract training hyperparameters
        training_config = config.get('training', {})

        self.lr = training_config.get('lr', 0.001)

        # Validate hyperparameters
        if self.lr <= 0:
            raise ValueError(f"training.lr must be > 0, got {self.lr}")

        # Class weights for handling imbalanced data (optional)
        # Format: [weight_for_negative, weight_for_positive]
        class_weights = training_config.get('class_weights', None)
        if class_weights is not None:
            if not isinstance(class_weights, (list, tuple)):
                raise TypeError(
                    f"training.class_weights must be a list or tuple, "
                    f"got {type(class_weights)}"
                )
            if len(class_weights) != 2:
                raise ValueError(
                    f"training.class_weights must have length 2, "
                    f"got {len(class_weights)}"
                )
            if any(w <= 0 for w in class_weights):
                raise ValueError(
                    "training.class_weights values must be positive, "
                    f"got {class_weights}"
                )
            # Use register_buffer for automatic device management
            self.register_buffer(
                'class_weights',
                torch.tensor(class_weights, dtype=torch.float32)
            )
        else:
            self.class_weights = None

        # Build binary classification head
        encoder_dim = self._get_encoder_dim()
        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)  # Binary classification
        )

        # Optimizer for global training
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.classifier.parameters()),
            lr=self.lr
        )

        print(f" Initialized JointTraining model:")
        print(f"  - Learning rate: {self.lr}")
        print(f"  - Class weights: {self.class_weights}")

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
        Forward pass through encoder and classifier.

        Args:
            x: Input features [N, input_dim]

        Returns:
            logits: Binary classification logits [N]
        """
        features = self.encoder(x)
        logits = self.classifier(features).squeeze(-1)
        return logits

    def train_episode(self, support: Dict[str, torch.Tensor],
                      query: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Train on a single episode (standard supervised learning).

        Note: Unlike meta-learning baselines, JointTraining does NOT use
        episodic training with support/query splits. However, to maintain
        interface compatibility with BaseComparisonModel, we implement this
        method by treating the combined support+query as a standard batch.

        Args:
            support: Support set data (merged with query for training)
            query: Query set data (merged with support for training)

        Returns:
            metrics: Dictionary containing loss and accuracy
        """
        self.train()

        # Merge support and query into a single batch
        # This is the correct behavior for Joint Training
        x_all = torch.cat([support['x'], query['x']], dim=0)
        y_all = torch.cat([support['y'], query['y']], dim=0)

        # Forward pass
        features = self.encoder(x_all)
        logits = self.classifier(features).squeeze(-1)

        # Compute binary cross-entropy loss
        if self.class_weights is not None:
            # Apply class weights
            weights = torch.where(
                y_all == 1,
                self.class_weights[1],
                self.class_weights[0]
            )
            loss = F.binary_cross_entropy_with_logits(
                logits, y_all.float(), weight=weights
            )
        else:
            loss = F.binary_cross_entropy_with_logits(logits, y_all.float())

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Compute metrics
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            labels = y_all.long()

            acc = (preds == labels).float().mean().item()

        return {
            'loss': loss.item(),
            'accuracy': acc,
            'batch_size': len(y_all)
        }

    def adapt(self, support: Dict[str, torch.Tensor]) -> None:
        """
        Adapt model to a new task (no-op for Joint Training).

        Joint Training is a zero-shot baseline that does NOT perform
        task-specific adaptation. This method is a no-op to maintain
        interface compatibility.

        Args:
            support: Support set for adaptation (unused)
        """
        # No adaptation needed for Joint Training
        # The global model is directly applied to test data
        self.eval()

        print(f" JointTraining: No adaptation needed (zero-shot baseline)")
        print(f"  - Support size: {len(support['y'])} (unused)")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict anomaly scores for input samples.

        Uses the globally trained model without any task-specific adaptation.

        Args:
            x: Input features [N, ...]

        Returns:
            scores: Anomaly scores in [0, 1] [N]
        """
        self.eval()

        with torch.no_grad():
            features = self.encoder(x)
            logits = self.classifier(features).squeeze(-1)
            scores = torch.sigmoid(logits)

        return scores
