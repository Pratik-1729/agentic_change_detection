import numpy as np

from app.interfaces.base_model import BaseChangeDetectionModel


class DummyModel(BaseChangeDetectionModel):

    def load(self):
        print("Dummy model loaded")

    def preprocess(self, img1, img2):
        return img1, img2

    def predict(self, img1, img2):
        return np.zeros((512, 512), dtype=np.float32)

    def postprocess(self, prediction):
        return prediction > 0.5