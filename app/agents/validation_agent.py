from app.interfaces.base_agent import BaseAgent
from app.models.vlm_factory import VLMFactory
from app.config.vlm_config import DEFAULT_VLM, get_vlm_config
from app.config.settings import VLM_MAX_WORKERS, VLM_RETRIES, VLM_TIMEOUT
from app.prompts.validation_prompt import VALIDATION_PROMPT
from app.utils.validation_parser import parse_validation_result
from app.utils.vlm_call import generate_with_retry, run_parallel
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

            def validate(region):
                prompt = VALIDATION_PROMPT.format(description=region["description"])
            
                text = generate_with_retry(
                    vlm, image=region["crop"], prompt=prompt,
                    max_new_tokens=150,
                    retries=VLM_RETRIES, timeout=VLM_TIMEOUT,
                )
            
                return {
                    **region,
                    "validation": text,
                    **parse_validation_result(text),
                }
            
            logger.info(
                f"Validating {len(state.descriptions)} region(s) "
                f"(max_workers={VLM_MAX_WORKERS})"
            )
            results = run_parallel(validate, state.descriptions, max_workers=VLM_MAX_WORKERS)
            
            validated = []
            for region, result in zip(state.descriptions, results):
                if isinstance(result, Exception):
                    logger.error(f"Region {region['id']} validation failed: {result}")
                    validated.append({
                        **region,
                        "validation": None,
                        "decision": "VALIDATION_FAILED",
                        "reason": str(result),
                        "confidence": None,
                    })
                    continue
                validated.append(result)
            
            state.validated_regions = validated
            
        except Exception as e:
            logger.error(f"Validation stage failed: {e}")
            state.errors.append(f"ValidationAgent: {e}")
            
        return state
