"""Puzzle-embedding table (inference path only).

Mirrors ``CastedSparseEmbedding`` from the TRM/HRM codebase.  Only the eval path
is needed: a plain lookup into the persistent ``weights`` buffer.  The non-
persistent training buffers (``local_weights``/``local_ids``) are omitted because
they are not part of the saved checkpoint.
"""
import torch
from torch import nn

from .common import trunc_normal_init_


class CastedSparseEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, batch_size: int, init_std: float, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to
        self.register_buffer(
            "weights",
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std),
            persistent=True,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Inference: simple gather, no gradient bookkeeping.
        return self.weights[inputs].to(self.cast_to)
