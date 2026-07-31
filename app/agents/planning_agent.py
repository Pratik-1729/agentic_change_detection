from app.interfaces.base_agent import BaseAgent
from app.registry.model_registry import ModelRegistry
from app.core.logger import logger

DEFAULT_MODEL = "changeformer"


class PlanningAgent(BaseAgent):


    def run(self, state):
        requested = (state.selected_model or DEFAULT_MODEL).lower()

        available = ModelRegistry.available_models()

        if requested not in available:
            msg = f"Unknown model '{requested}'. Available: {available}"
            logger.error(msg)
            state.errors.append(f"PlanningAgent: {msg}")
            return state

        state.selected_model = requested
        logger.info(f"Selected model: {state.selected_model}")

        return state
