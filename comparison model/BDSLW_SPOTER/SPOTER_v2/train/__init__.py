"""Training package initialization for SPOTER v2."""

from .train_spoter_v2_simple import train
from .multimodal_spoter import MultimodalSPOTERModel

__all__ = [
    "train",
    "MultimodalSPOTERModel",
]
