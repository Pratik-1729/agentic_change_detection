import uuid

from app.interfaces.base_agent import BaseAgent
from app.schemas.pipeline_state import PipelineState
from app.core.logger import logger


class IngestionAgent(BaseAgent):

    def run(self, state: PipelineState) -> PipelineState:

        if not state.job_id:
            state.job_id = str(uuid.uuid4())
        logger.info(f"Created Job {state.job_id}")

        return state