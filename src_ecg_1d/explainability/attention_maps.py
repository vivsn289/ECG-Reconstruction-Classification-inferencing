# src_ecg_1d/explainability/attention_maps.py
#
# DEPRECATED — use attention_utils.py instead.
#
# This module contained an earlier version of aggregate_attention() that
# incorrectly assumed a single batch element in its normalization step.
# The canonical implementation is in attention_utils.py, which handles
# batches of arbitrary size correctly.
#
# This file is kept to avoid breaking any external references.

from src_ecg_1d.explainability.attention_utils import aggregate_attention  # noqa: F401

__all__ = ["aggregate_attention"]
