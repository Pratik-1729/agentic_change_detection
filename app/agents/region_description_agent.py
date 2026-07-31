import numpy as np
from PIL import Image

from app.interfaces.base_agent import BaseAgent
from app.models.vlm_factory import VLMFactory
from app.config.vlm_config import DEFAULT_VLM, get_vlm_config
from app.config.settings import VLM_MAX_WORKERS, VLM_RETRIES, VLM_TIMEOUT
from app.prompts.region_prompt import REGION_DESCRIPTION_PROMPT
from app.utils.vlm_call import generate_with_retry, run_parallel
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

            def describe(crop):
                image = crop["after"]
                if not isinstance(image, Image.Image):
                    image = Image.fromarray(np.asarray(image))
            
                text = generate_with_retry(
                    vlm, image=image, prompt=REGION_DESCRIPTION_PROMPT,
                    max_new_tokens=200,
                    retries=VLM_RETRIES, timeout=VLM_TIMEOUT,
                )
            
                return {
                    "id": crop["id"],
                    "bbox": crop["bbox"],
                    "crop": image,
                    "description": text,
                }
            
            logger.info(
                f"Describing {len(state.crops)} region(s) "
                f"(max_workers={VLM_MAX_WORKERS})"
            )
            results = run_parallel(describe, state.crops, max_workers=VLM_MAX_WORKERS)
            
            descriptions = []
            failures = []
            
            for crop, result in zip(state.crops, results):
                if isinstance(result, Exception):
                    logger.error(f"Region {crop['id']} description failed: {result}")
                    failures.append({"id": crop["id"], "error": str(result)})
                    continue
                descriptions.append(result)
            
            state.descriptions = descriptions
            if failures:
                state.metadata["description_failures"] = failures
            
        except Exception as e:
            logger.error(f"Region description stage failed: {e}")
            state.errors.append(f"RegionDescriptionAgent: {e}")
            
        return state
