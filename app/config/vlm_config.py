"""
Central place to configure available VLMs.
"""
from app.config.settings import DEVICE

DEFAULT_VLM = "qwen"

VLM_CONFIGS = {
    "qwen": {
        "model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
        "device": DEVICE,
    },
    "dummy": {},
}


def get_vlm_config(vlm_name: str) -> dict:
    return VLM_CONFIGS.get(vlm_name.lower(), {})
