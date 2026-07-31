from abc import ABC, abstractmethod

from app.schemas.pipeline_state import PipelineState


class BaseVLM(ABC):
    """
    Base interface for all pipeline agents.
    """
    @abstractmethod
    def load(self):
        pass
    
    @abstractmethod
    def generate(self,image,prompt, **kwargs):
        """
        Execute agent logic and return updated state.
        """
        pass