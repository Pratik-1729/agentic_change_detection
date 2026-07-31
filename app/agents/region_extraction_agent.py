from app.interfaces.base_agent import BaseAgent
from app.detection.region_extractor import RegionExtractor
from app.core.logger import logger


class RegionExtractionAgent(BaseAgent):
    """
    Turns state.change_mask (produced by InferenceAgent) into a list
    of ChangeRegion bounding boxes worth describing/validating.

    Replaces the old ChangeDetectionAgent, which combined "run the
    model" + "extract regions" into one class (and had a typo in its
    name). Running the model is now InferenceAgent's job; this agent
    only does the mask -> regions step, so it can sit cleanly after
    InferenceAgent in the pipeline.
    """

    def __init__(self, extractor=None):
        self.extractor = extractor or RegionExtractor()

    def run(self, state):
        try:
            if state.change_mask is None:
                raise ValueError(
                    "No change_mask available — did InferenceAgent run first?"
                )

            state.regions = self.extractor.extract(mask=state.change_mask)
            logger.info(f"Extracted {len(state.regions)} candidate regions")

        except Exception as e:
            logger.error(f"Region extraction failed: {e}")
            state.errors.append(f"RegionExtractionAgent: {e}")

        return state
