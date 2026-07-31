"""
Central place to configure available models. See original docstring
for how to add a new one -- unchanged.
"""
import os

from app.config.settings import DEVICE

MODEL_CONFIGS = {
    "changeformer": {
        # override with: set CHANGEFORMER_CHECKPOINT=C:\path\to\ckpt.pt
        "checkpoint": os.environ.get(
            "CHANGEFORMER_CHECKPOINT",
            "weights/changeformer/best_ckpt.pt",
        ),
    },
    "dummy": {},
}


def get_model_config(model_name: str) -> dict:
    return MODEL_CONFIGS.get(model_name.lower(), {})
