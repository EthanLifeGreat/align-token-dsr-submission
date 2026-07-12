"""Utilities for external baseline reproduction workflows.

The modules in this package intentionally contain no dataset-specific paths,
audio, prompts, model weights, or trained checkpoints.
"""

from .common import (
    COSYVOICE_MODEL_ID,
    COSYVOICE_MODEL_REVISION,
    COSYVOICE_SOURCE_REVISION,
    SEED_VC_MODEL_ID,
    SEED_VC_MODEL_REVISION,
    SEED_VC_SOURCE_REVISION,
    WAV2VEC2_MODEL_ID,
    WAV2VEC2_MODEL_REVISION,
)

__all__ = [
    "COSYVOICE_MODEL_ID",
    "COSYVOICE_MODEL_REVISION",
    "COSYVOICE_SOURCE_REVISION",
    "SEED_VC_MODEL_ID",
    "SEED_VC_MODEL_REVISION",
    "SEED_VC_SOURCE_REVISION",
    "WAV2VEC2_MODEL_ID",
    "WAV2VEC2_MODEL_REVISION",
]
