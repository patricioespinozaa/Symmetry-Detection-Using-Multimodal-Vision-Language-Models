"""
molmo_model.py
--------------
Loads Molmo2 model and processor once per process and caches them.
Call get_model() from any module to reuse the loaded instance.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "allenai/Molmo2-8B"

_processor = None
_model = None


def get_model(model_id: str = MODEL_ID):
    """
    Load and return (processor, model).
    Subsequent calls within the same process return the cached instance.
    """
    global _processor, _model

    if _processor is None or _model is None:
        print(f"[molmo_model] Loading: {model_id}")

        _processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map="auto",
        )

        _model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        _model.eval()
        print("[molmo_model] Ready.")

    return _processor, _model
