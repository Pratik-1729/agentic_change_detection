from app.interfaces.base_agent import BaseAgent
from app.models.vlm_factory import VLMFactory
from app.config.vlm_config import DEFAULT_VLM, get_vlm_config
from app.prompts.validation_prompt import VALIDATION_PROMPT
from app.utils.validation_parser import parse_validation_result
from app.core.logger import logger


class ValidationAgent(BaseAgent):
    """
    Asks the VLM to judge whether each described region is a genuine
    change or a likely false positive (shadows, seasonal vegetation,
    misalignment, sensor noise, etc.)

    VLMFactory.create() here returns the SAME cached instance
    RegionDescriptionAgent already loaded (same vlm_name + kwargs),
    so this doesn't reload the model.
    """

    def run(self, state):
        try:
            if not state.descriptions:
                logger.info("No descriptions to validate — skipping")
                return state

            vlm_name = state.selected_vlm or DEFAULT_VLM
            vlm = VLMFactory.create(vlm_name, **get_vlm_config(vlm_name))

            validated = []

            for region in state.descriptions:
                prompt = VALIDATION_PROMPT.format(
                    description=region["description"]
                )

                result = vlm.generate(
                    image=region["crop"],
                    prompt=prompt,
                    max_new_tokens=150,
                )

                validated.append({
                    **region,
                    "validation": result,
                    **parse_validation_result(result),
                })

            state.validated_regions = validated

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            state.errors.append(f"ValidationAgent: {e}")

        return state
