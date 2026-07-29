from abc import ABC, abstractmethod
from typing import Any


class BaseChangeDetectionModel(ABC):

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def preprocess(self, image1: Any, image2: Any):
        ...

    @abstractmethod
    def predict(self, image1: Any, image2: Any):
        ...

    @abstractmethod
    def postprocess(self, prediction):
        ...