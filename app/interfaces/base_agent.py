from abc import ABC, abstractmethod

from app.schemas.pipeline_state import PipelineState


class BaseAgent(ABC):
    """
    Base interface for all pipeline agents.
    """

    @abstractmethod
    def run(self, state: PipelineState) -> PipelineState:
        """
        Execute agent logic and return updated state.
        """
        pass