# Original: torchtune/modules/loss/__init__.py
# Added extra loss classes 

from .ce_chunked_output_loss import CEWithChunkedOutputLoss
from .kd_losses import (
    ForwardKLLoss,
    ForwardKLWithChunkedOutputLoss,
    ReverseKLLoss,
    ReverseKLWithChunkedOutputLoss,
    SymmetricKLLoss,
    SymmetricKLWithChunkedOutputLoss,
    JSDistanceLoss,
    JSDistanceWithChunkedOutputLoss,
    TVDistanceLoss,
    TVDistanceWithChunkedOutputLoss,
)

__all__ = [
    "CEWithChunkedOutputLoss",
    "ForwardKLLoss",
    "ForwardKLWithChunkedOutputLoss",
    "ReverseKLLoss",
    "ReverseKLWithChunkedOutputLoss",
    "SymmetricKLLoss",
    "SymmetricKLWithChunkedOutputLoss",
    "JSDistanceLoss",
    "JSDistanceWithChunkedOutputLoss",
    "TVDistanceLoss",
    "TVDistanceWithChunkedOutputLoss",
]
