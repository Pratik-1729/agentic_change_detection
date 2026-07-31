from pathlib import Path

import numpy as np

from app.interfaces.base_agent import BaseAgent
from app.core.logger import logger

# NOTE: adjust this import to match your actual repo layout.
# I've assumed transforms.py lives at app/models/changeformer/transforms.py
# based on the files shared so far — move this import if it's elsewhere.
from app.models.changeformer.transforms import load_rgb_tif


def _load_image(path: str) -> np.ndarray:
    ext = Path(path).suffix.lower()

    if ext in (".tif", ".tiff"):
        return load_rgb_tif(path)

    from PIL import Image
    return np.array(Image.open(path).convert("RGB"))


class PreprocessingAgent(BaseAgent):


    def run(self, state):
        try:
            state.preprocessed_t1 = _load_image(state.image_t1)
            state.preprocessed_t2 = _load_image(state.image_t2)

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            state.errors.append(f"PreprocessingAgent: {e}")

        return state
