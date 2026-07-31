import numpy as np

from app.interfaces.base_model import BaseChangeDetectionModel
from app.registry.model_registry import ModelRegistry


class DummyModel(BaseChangeDetectionModel):
    """
    A trivial change detection 'model' that needs no checkpoint, no
    GPU, and no external repo. It returns a random binary mask shaped
    like the input.

    Useful for:
      - Verifying the pipeline wiring (ingestion -> preprocessing ->
        inference -> analysis -> report) end-to-end without waiting
        on a real model to load/run.
      - CI / unit tests that shouldn't depend on large checkpoints.
    """

    def load(self):
        pass

    def preprocess(self, image1, image2):
        return image1, image2

    def predict(self, image1, image2):
        h, w = np.array(image1).shape[:2]
        return np.random.randint(0, 2, size=(h, w), dtype=np.uint8)

    def postprocess(self, prediction):
        return prediction


ModelRegistry.register("dummy", DummyModel)
