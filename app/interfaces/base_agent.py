from abc import ABC, abstractmethod
from typing import Any


class BaseChangeDetectionModel(ABC):
    """
    Base interface for all change detection models.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None

    @abstractmethod
    def load(self) -> None:
        """Load pretrained weights."""
        ...

    @abstractmethod
    def preprocess(self, image1: Any, image2: Any):
        """Convert raw images into model input."""
        ...

    @abstractmethod
    def predict(self, image1: Any, image2: Any):
        """Run inference."""
        ...

    @abstractmethod
    def postprocess(self, prediction):
        """Convert raw model output into a binary mask."""
        ...