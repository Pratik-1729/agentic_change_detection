from app.interfaces.base_agent import BaseAgent
from app.detection.crop_extractor import CropExtractor
from app.core.logger import logger


class CropExtractionAgent(BaseAgent):
    """
    Crops the before/after images around each detected region so the
    VLM agents only look at the relevant area, not the whole scene.
    """

    def __init__(self, extractor=None):
        self.extractor = extractor or CropExtractor()

    def run(self, state):
        try:
            if not state.regions:
                logger.info("No regions to crop — skipping")
                return state

            state.crops = self.extractor.crop(
                before=state.preprocessed_t1,
                after=state.preprocessed_t2,
                regions=state.regions,
            )
            logger.info(f"Extracted {len(state.crops)} crops")

        except Exception as e:
            logger.error(f"Crop extraction failed: {e}")
            state.errors.append(f"CropExtractionAgent: {e}")

        return state
