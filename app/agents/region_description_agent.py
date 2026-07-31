import numpy as np
from PIL import Image

from app.interfaces.base_agent import BaseAgent
from app.models.vlm_factory import VLMFactory
from app.config.vlm_config import DEFAULT_VLM, get_vlm_config
from app.prompts.region_prompt import REGION_DESCRIPTION_PROMPT
from app.core.logger import logger


class RegionDescriptionAgent(BaseAgent):
    """
    Describes each cropped region using a VLM. Uses the "after" crop,
    since that reflects the region's current state.

    VLM selection follows the same pattern as change-detection models:
    state.selected_vlm picks it (falls back to DEFAULT_VLM), and
    VLMFactory.create() is cached so this agent and ValidationAgent
    share one loaded instance instead of loading the model twice.
    """

    def run(self, state):
        try:
            if not state.crops:
                logger.info("No crops to describe — skipping")
                return state

            vlm_name = state.selected_vlm or DEFAULT_VLM
            vlm = VLMFactory.create(vlm_name, **get_vlm_config(vlm_name))

            descriptions = []

            for i, crop in enumerate(state.crops):
                image = crop["after"]
                if not isinstance(image, Image.Image):
                    image = Image.fromarray(np.asarray(image))

                logger.info(f"Describing region {i + 1}/{len(state.crops)}")

                description = vlm.generate(
                    image=image,
                    prompt=REGION_DESCRIPTION_PROMPT,
                    max_new_tokens=200,
                )

                descriptions.append({
                    "id": crop["id"],
                    "bbox": crop["bbox"],
                    "crop": image,
                    "description": description,
                })

            state.descriptions = descriptions

        except Exception as e:
            logger.error(f"Region description failed: {e}")
            state.errors.append(f"RegionDescriptionAgent: {e}")

        return state
