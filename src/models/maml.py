"""
BASE_MAML: Model-Agnostic Meta-Learning for Fast Adaptation

Reference: TSA_MAML.md Section 2 (Standard MAML Review)

Core Idea:
- Learn a good initialization θ* that enables fast adaptation to new tasks
- Inner-loop: Task-specific adaptation using support set (few gradient steps)
- Outer-loop: Meta-update to improve the initialization

Technical Implementation:
- 2-level optimization (inner/outer loops)
- First-order approximation for efficiency
- Shared encoder with SABLE (transformer_b_small)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from .base import BaseComparisonModel
from . import register_model


@register_model("MAML")
class MAMLModel(BaseComparisonModel):
    """
    Standard MAML implementation for few-shot anomaly detection.

    Unified Constraints:
    - Encoder: Shared with SABLE (transformer_b_small)
    - Threshold: Search mode, F1 metric
    - Resampling: Disabled
    - Task construction: Episodic few-shot
    """

    def __init__(self, encoder: nn.Module, config: Dict[str, Any]):
        """
        Initialize MAML model.

        Args:
            encoder: Shared encoder (transformer_b_small)
            config: Model configuration from YAML
                - training.lr_inner: Inner-loop learning rate
                - training.lr_meta: Meta/outer-loop learning rate
                - training.inner_steps: Number of inner-loop gradient steps
        """
        super().__init__(encoder, config)

        # Extract MAML-specific hyperparameters
        training_config = config.get('training', {})
        self.lr_inner = training_config.get('lr_inner', 0.001)
        self.lr_meta = training_config.get('lr_meta', 0.001)
        self.inner_steps = training_config.get('inner_steps', 5)

        # Task-adapted parameters (set during adapt())
        self.adapted_classifier = None

        # Build classifier head (from encoder dim to binary output)
        encoder_dim = self._get_encoder_dim()
        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

        # Meta-optimizer for outer-loop update
        meta_params = list(self.encoder.parameters()) + list(self.classifier.parameters())
        self.meta_optimizer = torch.optim.Adam(meta_params, lr=self.lr_meta)

        print(f" Initialized MAML model:")
        print(f"  - Inner LR: {self.lr_inner}")
        print(f"  - Meta LR: {self.lr_meta}")
        print(f"  - Inner steps: {self.inner_steps}")

    def _get_encoder_dim(self) -> int:
        """Get encoder output dimension."""
        # Assuming encoder outputs a fixed-size feature vector
        # For transformer_b_small, this is typically 256 or 512
        # This should be determined dynamically based on actual encoder
        # For now, use a common default
        return 256  # TODO: Extract from actual encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder and classifier.

        Args:
            x: Input features

        Returns:
            logits: Binary classification logits [batch_size, 2]
                    Standard format for compatibility with FocalLoss
        """
        features = self.encoder(x)
        logits = self.classifier(features)  # [batch_size, 1]

        # Will convert binary class logits to standard 2-class format for FocalLoss compatibility
        # logits[:, 0] = negative class scores, logits[:, 1] = positive class scores
        logits_2class = torch.cat([-logits, logits], dim=1)  # [batch_size, 2]
        return logits_2class

    # ========= Functional forward helpers used for inner-loop =========

    def forward_with_weights(self, x: torch.Tensor, weights: dict) -> torch.Tensor:
        """
        Use specified weights for forward pass (key method for MAML inner loop).

        Args:
            x: inputfeature [N, input_dim]
            weights: parameter dictionary (parameter name -> parameter tensor)

        Returns:
            logits: modeloutput [N, 2]

        NOTE:
            This method is crucial for MAML! It must use the passed weights parameter for forward pass,
            rather than self.parameters(). Otherwise inner loop is completely ineffective!
        """
        return self._forward_with_params(x, weights)

    def _forward_with_params(self, x: torch.Tensor,
                             params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass using an explicit parameter dictionary.

        This keeps the inner-loop purely functional and avoids modifying
        self.parameters(), which is required for a correct MAML implementation.
        """
        # Encoder: we currently do not adapt encoder parameters in the
        # inner-loop. For simplicity and stability, we call self.encoder.
        features = self.encoder(x)

        # Classifier: use params dict for both Linear layers.
        # Expected classifier submodule structure:
        #   0: Linear(encoder_dim, 128)
        #   1: ReLU
        #   2: Dropout
        #   3: Linear(128, 1)
        w1 = params.get('classifier.0.weight', self.classifier[0].weight)
        b1 = params.get('classifier.0.bias', self.classifier[0].bias)
        w2 = params.get('classifier.3.weight', self.classifier[3].weight)
        b2 = params.get('classifier.3.bias', self.classifier[3].bias)

        x = F.linear(features, w1, b1)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        logits = F.linear(x, w2, b2)  # [batch_size, 1]

        # Convert to standard 2-class format for FocalLoss compatibility
        logits_2class = torch.cat([-logits, logits], dim=1)  # [batch_size, 2]
        return logits_2class

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute binary cross-entropy loss.

        Args:
            x: Input features
            y: Binary labels (0/1)

        Returns:
            loss: CE loss
        """
        logits = self.forward(x)  # [batch_size, 2]
        # Use standard cross-entropy loss (expects Long type label)
        loss = F.cross_entropy(logits, y.long())
        return loss

    def inner_loop_update(self, support: Dict[str, torch.Tensor],
                          num_steps: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Perform inner-loop adaptation on support set.

        This is the core of MAML: adapt model parameters to a specific task
        using a few gradient steps on the support set.

        Args:
            support: Support set data
                - 'x': Features [N_support, ...]
                - 'y': Labels [N_support]
            num_steps: Number of gradient steps (if None, use self.inner_steps)

        Returns:
            adapted_params: Task-adapted parameters (state dict)
        """
        if num_steps is None:
            num_steps = self.inner_steps

        # First-order MAML on a *separate* parameter copy.
        # We currently adapt only the classifier parameters; encoder
        # stays fixed to reduce computational cost and keep behaviour stable.
        params = {
            name: p.clone().detach().requires_grad_(True)
            for name, p in self.named_parameters()
            if name.startswith('classifier.')
        }

        x_sup = support['x']
        y_sup = support['y']

        for _ in range(num_steps):
            logits = self._forward_with_params(x_sup, params)  # [batch_size, 2]
            loss = F.cross_entropy(logits, y_sup.long())

            grads = torch.autograd.grad(
                loss,
                list(params.values()),
                create_graph=False,
                allow_unused=True,
            )

            new_params: Dict[str, torch.Tensor] = {}
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
        Train on a single episode (meta-training).

        MAML meta-training procedure:
        1. Sample a task (support + query)
        2. Inner-loop: Adapt to task using support set
        3. Outer-loop: Compute meta-gradient on query set
        4. Update initialization parameters

        Args:
            support: Support set for adaptation
            query: Query set for meta-gradient

        Returns:
            metrics: Dictionary containing loss and other metrics
        """
        self.train()

        # Cache original classifier parameters for restoration.
        original_classifier = {
            name: p.clone()
            for name, p in self.named_parameters()
            if name.startswith('classifier.')
        }

        # Step 1: inner-loop adaptation (functional) on support set.
        adapted_params = self.inner_loop_update(support)

        # Step 2: temporarily load adapted classifier params
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in adapted_params:
                    param.copy_(adapted_params[name])

        # Step 3: compute query loss while parameters are in adapted state
        query_loss = self.compute_loss(query['x'], query['y'])

        # Step 4: meta-gradient on initialization parameters
        self.meta_optimizer.zero_grad()
        query_loss.backward()

        # Step 5: restore original classifier weights (encoder unchanged)
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in original_classifier:
                    param.copy_(original_classifier[name])

        # Step 6: apply meta-update
        self.meta_optimizer.step()

        # Evaluate metrics on query set using *adapted* parameters for
        # prediction, to be consistent with the MAML evaluation protocol.
        with torch.no_grad():
            # Load adapted params again
            for name, param in self.named_parameters():
                if name in adapted_params:
                    param.copy_(adapted_params[name])

            logits = self.forward(query['x'])  # [batch_size, 2]
            # Use softmax to get probabilities, extract positive class (index 1)
            probs = F.softmax(logits, dim=1)[:, 1]  # [batch_size]
            preds = (probs >= 0.5).long()
            labels = query['y'].long()

            acc = (preds == labels).float().mean().item()

            # Restore initialization parameters for subsequent episodes
            for name, param in self.named_parameters():
                if name in original_classifier:
                    param.copy_(original_classifier[name])

        return {
            'loss': query_loss.item(),
            'accuracy': acc,
            'query_size': len(query['y'])
        }

    def adapt(self, support: Dict[str, torch.Tensor]) -> None:
        """
        Adapt model to a new task using support set (meta-test).

        During meta-test, we:
        1. Start from learned initialization θ*
        2. Perform inner-loop adaptation on support set
        3. Use adapted parameters for inference

        Args:
            support: Support set for adaptation
                - 'x': Features [N_support, ...]
                - 'y': Labels [N_support]
        """
        self.eval()

        # Perform inner-loop adaptation (no gradient tracking required here).
        with torch.no_grad():
            adapted_params = self.inner_loop_update(support)

        # Only store classifier parameters as adapted state.
        self.adapted_classifier = {
            name: p.clone() for name, p in adapted_params.items()
        }

        print(f" Adapted MAML to new task using {len(support['y'])} support samples")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict anomaly scores for input samples.

        If model has been adapted (via adapt()), use adapted parameters.
        Otherwise, use the meta-learned initialization.

        Args:
            x: Input features [N, ...]

        Returns:
            scores: Anomaly scores in [0, 1] [N]
        """
        self.eval()

        with torch.no_grad():
            # If adapted, temporarily load adapted parameters
            if self.adapted_classifier is not None:
                original_params = {
                    name: param.clone()
                    for name, param in self.named_parameters()
                    if name.startswith('classifier.')
                }

                for name, param in self.named_parameters():
                    if name in self.adapted_classifier:
                        param.copy_(self.adapted_classifier[name])

            # Forward pass
            logits = self.forward(x)  # [batch_size, 2]
            # Extract positive class (index 1) logit and convert to probability
            pos_logits = logits[:, 1]  # [batch_size]
            scores = torch.sigmoid(pos_logits)

            # Restore original classifier parameters if adapted
            if self.adapted_classifier is not None:
                for name, param in self.named_parameters():
                    if name in original_params:
                        param.copy_(original_params[name])

        return scores
