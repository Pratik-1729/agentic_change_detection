from app.models.model_factory import ModelFactory
from app.config.model_config import get_model_config
from app.config.settings import DEVICE
from app.interfaces.base_agent import BaseAgent
from app.core.logger import logger


class InferenceAgent(BaseAgent):

    def run(self, state):
        try:
            config = get_model_config(state.selected_model)

            model = ModelFactory.create(
                state.selected_model,
                device=DEVICE,
                **config,
            )

            img1, img2 = model.preprocess(
                state.preprocessed_t1,
                state.preprocessed_t2,
            )

            prediction = model.predict(img1, img2)

            state.change_mask = model.postprocess(prediction)

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            state.errors.append(f"InferenceAgent: {e}")

        return state
