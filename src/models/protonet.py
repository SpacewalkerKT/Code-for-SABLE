"""
BASE_ProtoNet: Prototypical Networks for Few-Shot Learning

Reference: ProtoNet.md (Experiment reports/for comparison model research/)

Core Idea:
- Each class is represented by the mean of its support sample embeddings (Prototype)
- Query samples are classified by distance to each class prototype
- Use squared Euclidean distance + softmax over negative distances

Technical Implementation:
- Metric-based few-shot classification
- Class prototypes: c_k = mean(support embeddings for class k)
- Distance-based classification
- Shared encoder with SABLE (transformer_b_small)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

from .base import BaseComparisonModel
from . import register_model


@register_model("ProtoNet")
class ProtoNetModel(BaseComparisonModel):
    """
    Prototypical Networks for few-shot anomaly detection.

    Unified Constraints:
    - Encoder: Shared with SABLE (transformer_b_small)
    - Threshold: Search mode, F1 metric
    - Resampling: Disabled
    - Task construction: Episodic few-shot (2-way for binary classification)
    """

    def __init__(self, encoder: nn.Module, config: Dict[str, Any]):
        """
        Initialize ProtoNet model.

        Args:
            encoder: Shared encoder (transformer_b_small)
            config: Model configuration from YAML
                - model.embedding_dim: Embedding dimension (default: 256)
                - model.distance: Distance metric (default: 'euclidean')
                - training.lr: Learning rate for encoder
        """
        super().__init__(encoder, config)

        # Extract ProtoNet-specific hyperparameters
        model_config = config.get('model', {})

        # Get encoder output dimension dynamically
        self.embedding_dim = self._get_encoder_dim()
        config_dim = model_config.get('embedding_dim', None)
        if config_dim is not None and config_dim != self.embedding_dim:
            print(f"  Warning: Config embedding_dim={config_dim} differs from "
                  f"actual encoder output dim={self.embedding_dim}. Using actual dim.")

        self.distance_metric = model_config.get('distance', 'euclidean')

        # Verify distance metric
        if self.distance_metric != 'euclidean':
            raise ValueError(
                f"ProtoNet currently only supports 'euclidean' distance, "
                f"got '{self.distance_metric}'"
            )

        # Task-adapted prototypes (set during adapt())
        self.prototypes: Optional[Dict[int, torch.Tensor]] = None

        # Build classifier head for standard forward pass (compatibility with SABLE training loop)
        # Note: ProtoNet is primarily metric-based, but we add a classifier for
        # compatibility with standard PyTorch training loops
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)  # Binary classification
        )

        # Optimizer for encoder + classifier (ProtoNet trains end-to-end)
        training_config = config.get('training', {})
        lr = training_config.get('lr', 0.001)
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.classifier.parameters()),
            lr=lr
        )

        print(f" Initialized ProtoNet model:")
        print(f"  - Embedding dim: {self.embedding_dim}")
        print(f"  - Distance metric: {self.distance_metric}")
        print(f"  - Learning rate: {lr}")

    def _get_encoder_dim(self) -> int:
        """
        Dynamically get encoder output dimension.

        Returns:
            int: Encoder output dimension
        """
        try:
            # Create a small dummy input to probe encoder output shape
            # Assuming input shape from SABLE config (adjust if needed)
            device = next(self.encoder.parameters()).device
            dummy_input = torch.randn(1, 32, device=device)  # [batch=1, seq_len=32]

            with torch.no_grad():
                dummy_output = self.encoder(dummy_input)

            # Return last dimension (embedding dim)
            return dummy_output.shape[-1]
        except Exception as e:
            print(f"  Warning: Could not probe encoder output dim ({e}). "
                  f"Falling back to config default.")
            return self.config.get('model', {}).get('embedding_dim', 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder and classifier.

        This method provides compatibility with standard PyTorch training loops
        (e.g., SABLE training loop). For metric-based few-shot learning,
        use train_episode() and adapt() methods instead.

        Args:
            x: Input features [N, input_dim]

        Returns:
            logits: Binary classification logits [N, 2]
                    Standard format for compatibility with FocalLoss

        Note:
            This forward() method uses a learned classifier head, which is
            different from ProtoNet's metric-based approach (using prototypes).
            The classifier is trained end-to-end with the encoder.
        """
        features = self.encoder(x)
        logits = self.classifier(features)  # [N, 1]

        # Convert to standard 2-class format for FocalLoss compatibility
        logits_2class = torch.cat([-logits, logits], dim=1)  # [N, 2]
        return logits_2class

    def compute_prototypes(self, support: Dict[str, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Calculate class prototypes (mean of support sample embeddings)

        For each class k, prototype is defined as:
        c_k = (1/|S_k|) * Σ_{x_i ∈ S_k} f_φ(x_i)

        Args:
            support: Support set data
                - 'x': Features [N_support, ...]
                - 'y': Labels [N_support] (0 or 1 for binary classification)

        Returns:
            prototypes: Dict mapping class label -> prototype vector
                {0: tensor[embedding_dim], 1: tensor[embedding_dim]}
        """
        x_support = support['x']
        y_support = support['y']

        # Get embeddings for all support samples
        with torch.set_grad_enabled(self.training):
            embeddings = self.encoder(x_support)  # [N_support, embedding_dim]

        # Compute prototype for each class
        prototypes = {}
        unique_labels = torch.unique(y_support)

        for label in unique_labels:
            label_int = int(label.item())
            # Get embeddings for this class
            class_mask = (y_support == label)
            class_embeddings = embeddings[class_mask]  # [N_class, embedding_dim]

            # Check for empty class (should not happen in proper episodic sampling)
            if class_embeddings.numel() == 0:
                raise ValueError(
                    f"Class {label_int} has no samples in support set. "
                    "Cannot compute prototype."
                )

            # Compute mean (prototype)
            prototype = class_embeddings.mean(dim=0)  # [embedding_dim]
            prototypes[label_int] = prototype

        return prototypes

    def compute_distances(self, query_embeddings: torch.Tensor,
                         prototypes: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Calculate squared Euclidean distance from query samples to each class prototype

        Distance definition: d(z, c_k) = ||z - c_k||_2^2

        Args:
            query_embeddings: Query embeddings [N_query, embedding_dim]
            prototypes: Class prototypes {class_id: prototype_vector}

        Returns:
            distances: Distance matrix [N_query, num_classes]
                distances[i, k] = ||z_i - c_k||^2
        """
        # Stack prototypes in sorted order by class label
        sorted_labels = sorted(prototypes.keys())
        prototype_tensor = torch.stack([prototypes[k] for k in sorted_labels])
        # prototype_tensor: [num_classes, embedding_dim]

        # Compute pairwise squared Euclidean distances
        # ||z - c||^2 = ||z||^2 + ||c||^2 - 2*<z, c>
        query_norm = (query_embeddings ** 2).sum(dim=1, keepdim=True)
        # [N_query, 1]
        proto_norm = (prototype_tensor ** 2).sum(dim=1, keepdim=True).t()
        # [1, num_classes]
        cross_term = torch.matmul(query_embeddings, prototype_tensor.t())
        # [N_query, num_classes]

        distances = query_norm + proto_norm - 2 * cross_term
        # [N_query, num_classes]

        return distances

    def compute_logits(self, query_embeddings: torch.Tensor,
                      prototypes: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Calculate classification logits for query samples (negative distances)

        Classification probability: p(y=k|x) = exp(-d(f(x), c_k)) / Σ_k' exp(-d(f(x), c_k'))

        Args:
            query_embeddings: Query embeddings [N_query, embedding_dim]
            prototypes: Class prototypes

        Returns:
            logits: Negative distances [N_query, num_classes]
        """
        distances = self.compute_distances(query_embeddings, prototypes)
        # Use negative distances as logits
        logits = -distances
        return logits

    def train_episode(self, support: Dict[str, torch.Tensor],
                      query: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Train on a single episode (episodic training).

        ProtoNet episodic training:
        1. Compute class prototypes from support set
        2. Compute distances from query samples to prototypes
        3. Compute cross-entropy loss on query set
        4. Update encoder parameters

        Args:
            support: Support set for computing prototypes
            query: Query set for loss computation

        Returns:
            metrics: Dictionary containing loss and accuracy
        """
        self.train()

        # Step 1: Compute prototypes from support set
        prototypes = self.compute_prototypes(support)

        # Step 2: Get query embeddings
        x_query = query['x']
        y_query = query['y']

        query_embeddings = self.encoder(x_query)  # [N_query, embedding_dim]

        # Step 3: Compute logits (negative distances)
        logits = self.compute_logits(query_embeddings, prototypes)
        # [N_query, num_classes]

        # Step 4: Compute cross-entropy loss
        # Need to convert class labels to indices for cross_entropy
        sorted_labels = sorted(prototypes.keys())

        # Validate: ProtoNet requires non-negative integer labels
        if min(sorted_labels) < 0:
            raise ValueError(
                f"ProtoNet only supports non-negative labels, got {sorted_labels}. "
                "Negative labels will cause incorrect tensor indexing."
            )

        # Create label mapping tensor (vectorized, no GPU->CPU sync)
        # This avoids the performance issue of calling .item() in a loop
        max_label = max(sorted_labels)
        label_map = torch.full(
            (max_label + 1,), -1, dtype=torch.long, device=y_query.device
        )
        for idx, label in enumerate(sorted_labels):
            label_map[label] = idx

        # Convert query labels to indices using tensor indexing
        y_query_idx = label_map[y_query.long()]

        # Validate: check for unmapped labels
        if (y_query_idx == -1).any():
            raise ValueError(
                f"Found unmapped labels in query set. Expected {sorted_labels}"
            )

        loss = F.cross_entropy(logits, y_query_idx)

        # Step 5: Backward and optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Compute accuracy
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y_query_idx).float().mean().item()

        return {
            'loss': loss.item(),
            'accuracy': acc,
            'query_size': len(y_query)
        }

    def adapt(self, support: Dict[str, torch.Tensor]) -> None:
        """
        Adapt model to a new task by computing prototypes (meta-test).

        During meta-test:
        1. Compute class prototypes from support set
        2. Store prototypes for inference

        Args:
            support: Support set for adaptation
                - 'x': Features [N_support, ...]
                - 'y': Labels [N_support]
        """
        self.eval()

        with torch.no_grad():
            # Compute and store prototypes
            self.prototypes = self.compute_prototypes(support)

        num_classes = len(self.prototypes)
        support_per_class = {
            label: (support['y'] == label).sum().item()
            for label in self.prototypes.keys()
        }

        print(f" Adapted ProtoNet to new task:")
        print(f"  - Num classes: {num_classes}")
        print(f"  - Support per class: {support_per_class}")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict anomaly scores for input samples.

        For binary classification:
        - Compute distances to both prototypes
        - Return probability of higher label (assumed to be anomaly class)
        - Supports any non-negative integer label pair (e.g., [0,1], [2,3], etc.)

        Args:
            x: Input features [N, ...]

        Returns:
            scores: Anomaly scores (probability of higher label) in [0, 1] [N]
        """
        self.eval()

        if self.prototypes is None:
            raise RuntimeError(
                "Model has not been adapted yet. "
                "Call adapt() before predict()."
            )

        with torch.no_grad():
            # Get query embeddings
            embeddings = self.encoder(x)  # [N, embedding_dim]

            # Compute logits
            logits = self.compute_logits(embeddings, self.prototypes)
            # [N, num_classes]

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)  # [N, num_classes]

            # For binary classification, return P(higher label)
            # In sorted order, index 1 is the higher label (assumed anomaly)
            sorted_labels = sorted(self.prototypes.keys())

            if len(sorted_labels) == 2:
                # Binary case: return probability of higher label (assumed anomaly)
                # Use index position instead of assuming specific label value
                scores = probs[:, 1]
            else:
                # Multi-class case: use max probability as confidence
                # (not typical for anomaly detection)
                scores = probs.max(dim=1)[0]

        return scores
