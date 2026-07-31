import numpy as np

from app.interfaces.base_agent import BaseAgent
from app.core.logger import logger


class AnalysisAgent(BaseAgent):
    """
    Computes summary statistics from whatever the pipeline produced:
    overall changed area (from change_mask), region counts, and -- if
    validation ran -- how many regions were judged genuine changes vs
    false positives (using ValidationAgent's parsed "decision" field,
    not raw text matching).
    """

    def run(self, state):
        try:
            stats = {}

            if state.change_mask is not None:
                mask = np.asarray(state.change_mask)
                changed_pixels = int(np.sum(mask > 0))
                total_pixels = int(mask.size)

                stats["image_shape"] = list(mask.shape)
                stats["changed_pixels"] = changed_pixels
                stats["total_pixels"] = total_pixels
                stats["percent_area_changed"] = (
                    round(100 * changed_pixels / total_pixels, 2)
                    if total_pixels else 0.0
                )

            stats["num_regions_detected"] = len(state.regions)
            stats["num_regions_described"] = len(state.descriptions)

            if state.validated_regions:
                true_change = sum(
                    1 for r in state.validated_regions
                    if r.get("decision") == "TRUE_CHANGE"
                )
                false_positive = sum(
                    1 for r in state.validated_regions
                    if r.get("decision") == "FALSE_POSITIVE"
                )
                unclear = len(state.validated_regions) - true_change - false_positive

                confidences = [
                    r["confidence"] for r in state.validated_regions
                    if r.get("confidence") is not None
                ]

                stats["num_true_change"] = true_change
                stats["num_false_positive"] = false_positive
                stats["num_unclear_validation"] = unclear
                if confidences:
                    stats["avg_validation_confidence"] = round(
                        sum(confidences) / len(confidences), 1
                    )

            state.statistics = stats
            logger.info(f"Analysis complete: {stats}")

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            state.errors.append(f"AnalysisAgent: {e}")

        return state
