from pathlib import Path

from PIL import Image

from app.agents.base_agent import BaseAgent
from app.schemas.pipeline_state import PipelineState


class ValidationAgent(BaseAgent):

    def run(self, state: PipelineState):

        if not Path(state.image_t1).exists():
            state.errors.append("Image T1 not found")

        if not Path(state.image_t2).exists():
            state.errors.append("Image T2 not found")

        if state.errors:
            return state

        img1 = Image.open(state.image_t1)
        img2 = Image.open(state.image_t2)

        if img1.size != img2.size:
            state.errors.append("Image dimensions do not match")
            return state

        state.validation_status = True

        return state