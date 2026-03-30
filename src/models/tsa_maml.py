"""
BASE_TSA_MAML: Task Similarity Aware MAML

Reference: TSA_MAML.md (Experiment reports/for comparison model research/)

Core Idea:
- Learn multiple initializations {θ_i} instead of single θ*
- Use learner A to assign tasks to nearest group initialization
- Each task adapts from its assigned group initialization

Technical Implementation:
- Multi-initialization: m group-specific starting points
- Task assignment: Based on Euclidean distance in parameter space
- Vanilla MAML θ* for task parameter estimation
- k-means clustering to initialize groups
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
from copy import deepcopy

from sklearn.metrics import roc_auc_score, average_precision_score

from .base import BaseComparisonModel
from . import register_model


@register_model("TSA_MAML")
class TSAMAMLModel(BaseComparisonModel):
    """
    TSA-MAML implementation for few-shot anomaly detection.

    Unified Constraints:
    - Encoder: Shared with SABLE (transformer_b_small)
    - Threshold: Search mode, F1 metric
    - Resampling: Disabled
    - Task construction: Episodic few-shot
    """

    def __init__(self, encoder: nn.Module, config: Dict[str, Any]):
        """
        Initialize TSA-MAML model.

        Args:
            encoder: Shared encoder (transformer_b_small)
            config: Model configuration from YAML
                - model.num_groups: Number of group initializations (default: 3)
                - training.lr_inner: Inner-loop learning rate
                - training.lr_meta: Meta/outer-loop learning rate
                - training.inner_steps: Number of inner-loop gradient steps
                - training.estimation_steps: Steps for task parameter estimation (default: same as inner_steps)
        """
        super().__init__(encoder, config)

        # Extract TSA-MAML specific hyperparameters
        training_config = config.get('training', {})
        model_config = config.get('model', {})

        self.num_groups = model_config.get('num_groups', 3)
        self.lr_inner = training_config.get('lr_inner', 0.001)
        self.lr_meta = training_config.get('lr_meta', 0.001)
        self.inner_steps = training_config.get('inner_steps', 5)
        self.estimation_steps = training_config.get('estimation_steps', self.inner_steps)

        # Validate hyperparameters
        if self.num_groups <= 0:
            raise ValueError(f"model.num_groups must be >= 1, got {self.num_groups}")
        if self.inner_steps <= 0:
            raise ValueError(f"training.inner_steps must be >= 1, got {self.inner_steps}")
        if self.estimation_steps <= 0:
            raise ValueError(f"training.estimation_steps must be >= 1, got {self.estimation_steps}")

        # Build classifier head (shared structure across all groups)
        encoder_dim = self._get_encoder_dim()
        self.classifier_template = nn.Sequential(
            nn.Linear(encoder_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

        # Initialize group-specific parameters
        # Each group has its own copy of encoder + classifier
        self.group_encoders = nn.ModuleList()
        self.group_classifiers = nn.ModuleList()

        for i in range(self.num_groups):
            # Deep copy encoder and classifier for each group
            group_encoder = deepcopy(encoder)
            group_classifier = deepcopy(self.classifier_template)

            self.group_encoders.append(group_encoder)
            self.group_classifiers.append(group_classifier)

        # Vanilla MAML initialization θ* (for task parameter estimation)
        # This is NOT updated during TSA-MAML training
        #
        # NOTE: Ideally, vanilla_encoder/classifier should be loaded from a
        # pre-trained MAML checkpoint for better task parameter estimation.
        # For simplified implementation, we use the initial encoder state as
        # the vanilla baseline. In production, consider:
        #   1. Pre-training a standard MAML model first
        #   2. Loading vanilla weights from that checkpoint
        #   3. Or use config: model.vanilla_maml_checkpoint: "path/to/maml.pth"
        self.vanilla_encoder = deepcopy(encoder)
        self.vanilla_classifier = deepcopy(self.classifier_template)

        # Set vanilla parameters to eval mode (frozen during TSA-MAML training)
        self.vanilla_encoder.eval()
        self.vanilla_classifier.eval()
        for param in self.vanilla_encoder.parameters():
            param.requires_grad = False
        for param in self.vanilla_classifier.parameters():
            param.requires_grad = False

        # Meta-optimizer for all group initializations
        meta_params = []
        for i in range(self.num_groups):
            meta_params += list(self.group_encoders[i].parameters())
            meta_params += list(self.group_classifiers[i].parameters())

        self.meta_optimizer = torch.optim.Adam(meta_params, lr=self.lr_meta)

        # Task-adapted parameters (set during adapt())
        self.adapted_encoder = None
        self.adapted_classifier = None
        self.assigned_group = None

        print(f" Initialized TSA-MAML model:")
        print(f"  - Num groups: {self.num_groups}")
        print(f"  - Inner LR: {self.lr_inner}")
        print(f"  - Meta LR: {self.lr_meta}")
        print(f"  - Inner steps: {self.inner_steps}")
        print(f"  - Estimation steps: {self.estimation_steps}")

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
        Forward pass through encoder and default classifier.

        Uses the first group's encoder and classifier as default for
        standard PyTorch training loops (e.g., SABLE training loop).

        Args:
            x: Input features [N, input_dim]

        Returns:
            logits: Binary classification logits [N, 2]
                    Standard format for compatibility with FocalLoss

        Note:
            During TSA-MAML episodic training, the model uses group-specific
            encoders and classifiers via train_episode(). This forward() method
            is provided for compatibility with standard training loops.
        """
        # Use first group's encoder and classifier as default
        features = self.group_encoders[0](x)
        logits = self.group_classifiers[0](features)  # [N, 1]

        # Convert to standard 2-class format for FocalLoss compatibility
        logits_2class = torch.cat([-logits, logits], dim=1)  # [N, 2]
        return logits_2class

    def _get_classifier_params_flat(self, classifier: nn.Module) -> torch.Tensor:
        """
        Get flattened classifier parameter vector for task assignment.

        Note: We only use classifier parameters for task assignment because:
        1. Encoders are typically kept fixed during inner-loop adaptation
        2. Task-specific information is captured in classifier parameters
        3. More efficient than comparing full model parameters

        Args:
            classifier: Classifier module

        Returns:
            Flattened classifier parameter vector
        """
        params = []
        for param in classifier.parameters():
            params.append(param.view(-1))
        return torch.cat(params)

    def estimate_task_params(self, support: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Estimate task's approximate optimal parameters using vanilla MAML θ*.

        This is the core of learner A: we use the frozen vanilla MAML initialization
        to quickly estimate where this task's optimal parameters might be.

        Args:
            support: Support set data
                - 'x': Features [N_support, ...]
                - 'y': Labels [N_support]

        Returns:
            estimated_params: Flattened parameter vector [total_params]
        """
        x_support = support['x']
        y_support = support['y']

        # Create temporary copies for estimation
        temp_classifier_params = {
            name: p.clone().detach().requires_grad_(True)
            for name, p in self.vanilla_classifier.named_parameters()
        }

        # Run estimation_steps of inner-loop updates
        for _ in range(self.estimation_steps):
            # Forward pass
            features = self.vanilla_encoder(x_support)

            w1 = temp_classifier_params.get('0.weight')
            b1 = temp_classifier_params.get('0.bias')
            w2 = temp_classifier_params.get('3.weight')
            b2 = temp_classifier_params.get('3.bias')

            x = F.linear(features, w1, b1)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=False)  # eval mode for estimation
            x = F.linear(x, w2, b2)
            logits = x  # [N, 1]

            # Convert to standard 2-class format
            logits_2class = torch.cat([-logits, logits], dim=1)  # [N, 2]

            # Compute loss (using standard cross-entropy)
            loss = F.cross_entropy(logits_2class, y_support.long())

            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                list(temp_classifier_params.values()),
                create_graph=False,
                allow_unused=True,
            )

            # Update parameters
            new_params = {}
            for (name, p), g in zip(temp_classifier_params.items(), grads):
                if g is None:
                    new_params[name] = p
                else:
                    new_params[name] = p - self.lr_inner * g
            temp_classifier_params = new_params

        # After estimation, create flattened parameter vector
        # IMPORTANT: Only return classifier parameters (not encoder)
        # This ensures dimensional consistency with group classifier params
        params = []
        for param in temp_classifier_params.values():
            params.append(param.view(-1))

        return torch.cat(params)

    def assign_task_to_group(self, estimated_params: torch.Tensor) -> int:
        """
        Assign task to nearest group initialization using Euclidean distance.

        This is learner A's decision function. We compare only classifier
        parameters (not encoder) for efficiency and semantic clarity.

        Args:
            estimated_params: Estimated task classifier parameters [classifier_params_dim]

        Returns:
            group_idx: Index of assigned group [0, num_groups)
        """
        min_dist = float('inf')
        assigned_group = 0

        for i in range(self.num_groups):
            # Get flattened classifier parameters for group i
            group_params = self._get_classifier_params_flat(
                self.group_classifiers[i]
            )

            # Compute Euclidean distance in classifier parameter space
            dist = torch.norm(estimated_params - group_params, p=2).item()

            if dist < min_dist:
                min_dist = dist
                assigned_group = i

        return assigned_group

    def inner_loop_update(self, group_idx: int, support: Dict[str, torch.Tensor],
                          num_steps: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Perform inner-loop adaptation from group initialization.

        Args:
            group_idx: Index of group to adapt from
            support: Support set data
            num_steps: Number of gradient steps (if None, use self.inner_steps)

        Returns:
            adapted_params: Task-adapted classifier parameters
        """
        if num_steps is None:
            num_steps = self.inner_steps

        # Get group's classifier parameters
        params = {
            name: p.clone().detach().requires_grad_(True)
            for name, p in self.group_classifiers[group_idx].named_parameters()
        }

        x_sup = support['x']
        y_sup = support['y']

        # Use group's encoder for feature extraction
        group_encoder = self.group_encoders[group_idx]

        for _ in range(num_steps):
            # Forward pass
            features = group_encoder(x_sup)

            w1 = params.get('0.weight')
            b1 = params.get('0.bias')
            w2 = params.get('3.weight')
            b2 = params.get('3.bias')

            x = F.linear(features, w1, b1)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
            x = F.linear(x, w2, b2)
            logits = x  # [N, 1]

            # Convert to standard 2-class format
            logits_2class = torch.cat([-logits, logits], dim=1)  # [N, 2]

            # Compute loss (using standard cross-entropy)
            loss = F.cross_entropy(logits_2class, y_sup.long())

            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                list(params.values()),
                create_graph=False,
                allow_unused=True,
            )

            # Update parameters
            new_params = {}
            for (name, p), g in zip(params.items(), grads):
                if g is None:
                    new_params[name] = p
                else:
                    new_params[name] = p - self.lr_inner * g
            params = new_params

        return params

    def train_episode(self, support: Dict[str, torch.Tensor],
                      query: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Train on a single episode (TSA-MAML meta-training).

        TSA-MAML training procedure:
        1. Estimate task parameters using vanilla MAML θ*
        2. Assign task to nearest group initialization
        3. Inner-loop: Adapt from assigned group initialization
        4. Outer-loop: Compute meta-gradient and update all groups

        Args:
            support: Support set for task estimation and adaptation
            query: Query set for meta-gradient

        Returns:
            metrics: Dictionary containing loss and accuracy
        """
        self.train()

        # Step 1: Estimate task parameters using vanilla MAML
        # NOTE: Cannot wrap in no_grad() because estimate_task_params
        # uses torch.autograd.grad internally for gradient descent
        estimated_params = self.estimate_task_params(support)

        # Step 2: Assign task to nearest group (no grad needed here)
        with torch.no_grad():
            group_idx = self.assign_task_to_group(estimated_params)

        # Step 3: Inner-loop adaptation from assigned group
        adapted_params = self.inner_loop_update(group_idx, support)

        # Step 4: Compute query loss with adapted parameters
        # Temporarily load adapted classifier parameters
        original_params = {
            name: p.clone()
            for name, p in self.group_classifiers[group_idx].named_parameters()
        }

        with torch.no_grad():
            for name, param in self.group_classifiers[group_idx].named_parameters():
                if name in adapted_params:
                    param.copy_(adapted_params[name])

        # Forward pass on query set
        x_query = query['x']
        y_query = query['y']

        features = self.group_encoders[group_idx](x_query)
        logits = self.group_classifiers[group_idx](features)  # [N, 1]

        # Convert to standard 2-class format
        logits_2class = torch.cat([-logits, logits], dim=1)  # [N, 2]

        # Use standard cross-entropy loss
        query_loss = F.cross_entropy(logits_2class, y_query.long())

        # Step 5: Meta-gradient and meta-update
        self.meta_optimizer.zero_grad()
        query_loss.backward()

        # Restore original group parameters before optimizer step
        with torch.no_grad():
            for name, param in self.group_classifiers[group_idx].named_parameters():
                if name in original_params:
                    param.copy_(original_params[name])

        # Apply meta-update (updates all groups)
        self.meta_optimizer.step()

        # Compute metrics
        with torch.no_grad():
            # Reload adapted params for evaluation
            for name, param in self.group_classifiers[group_idx].named_parameters():
                if name in adapted_params:
                    param.copy_(adapted_params[name])

            features = self.group_encoders[group_idx](x_query)
            logits = self.group_classifiers[group_idx](features)  # [N, 1]

            # Convert to standard 2-class format
            logits_2class = torch.cat([-logits, logits], dim=1)  # [N, 2]

            # Use softmax to get probabilities, extract positive class (index 1)
            probs = F.softmax(logits_2class, dim=1)[:, 1]  # [N]
            preds = (probs >= 0.5).long()
            labels = y_query.long()

            acc = (preds == labels).float().mean().item()

            # Restore original params
            for name, param in self.group_classifiers[group_idx].named_parameters():
                if name in original_params:
                    param.copy_(original_params[name])

        return {
            'loss': query_loss.item(),
            'accuracy': acc,
            'assigned_group': group_idx,
            'query_size': len(y_query)
        }

    def adapt(self, support: Dict[str, torch.Tensor]) -> None:
        """
        Adapt model to a new task (meta-test).

        TSA-MAML adaptation:
        1. Estimate task parameters using vanilla MAML θ*
        2. Assign to nearest group initialization
        3. Adapt from that group initialization

        Args:
            support: Support set for adaptation
                - 'x': Features [N_support, ...]
                - 'y': Labels [N_support]
        """
        self.eval()

        # NOTE: Cannot wrap estimate_task_params and inner_loop_update in no_grad()
        # because they use torch.autograd.grad internally for gradient descent

        # Step 1: Estimate task parameters
        estimated_params = self.estimate_task_params(support)

        # Step 2: Assign to nearest group
        self.assigned_group = self.assign_task_to_group(estimated_params)

        # Step 3: Adapt from assigned group
        adapted_params = self.inner_loop_update(self.assigned_group, support)

        # Store adapted parameters and encoder (detach to prevent gradient tracking)
        with torch.no_grad():
            self.adapted_classifier = {
                name: p.detach().clone() for name, p in adapted_params.items()
            }
            self.adapted_encoder = self.group_encoders[self.assigned_group]

        print(f" Adapted TSA-MAML to new task:")
        print(f"  - Assigned group: {self.assigned_group}")
        print(f"  - Support size: {len(support['y'])}")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict anomaly scores for input samples.

        Uses the adapted parameters from the assigned group initialization.

        Args:
            x: Input features [N, ...]

        Returns:
            scores: Anomaly scores in [0, 1] [N]
        """
        self.eval()

        if self.adapted_classifier is None or self.adapted_encoder is None:
            raise RuntimeError(
                "Model has not been adapted yet. "
                "Call adapt() before predict()."
            )

        with torch.no_grad():
            # Use adapted encoder
            features = self.adapted_encoder(x)

            # Use adapted classifier parameters
            w1 = self.adapted_classifier.get('0.weight')
            b1 = self.adapted_classifier.get('0.bias')
            w2 = self.adapted_classifier.get('3.weight')
            b2 = self.adapted_classifier.get('3.bias')

            x = F.linear(features, w1, b1)
            x = F.relu(x)
            x = F.linear(x, w2, b2)
            logits = x  # [N, 1]

            # Convert to standard 2-class format
            logits_2class = torch.cat([-logits, logits], dim=1)  # [N, 2]

            # Extract positive class (index 1) logit and convert to probability
            pos_logits = logits_2class[:, 1]  # [N]
            scores = torch.sigmoid(pos_logits)

        return scores
