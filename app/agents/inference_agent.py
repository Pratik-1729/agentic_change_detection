from app.models.model_factory import ModelFactory
from app.interfaces.base_agent import BaseAgent
class InferenceAgent(BaseAgent):

    def run(self, state):

        model = ModelFactory.create(
            state.selected_model,
            device="cpu"
        )

        img1, img2 = model.preprocess(
            state.preprocessed_t1,
            state.preprocessed_t2
        )

        prediction = model.predict(img1, img2)

        state.change_mask = model.postprocess(prediction)

        return state