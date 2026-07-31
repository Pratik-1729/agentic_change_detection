import app.models  # noqa: F401  (side-effect: registers every model + VLM)

from app.agents.ingestion_agent import IngestionAgent
from app.agents.preprocessing_agent import PreprocessingAgent
from app.agents.planning_agent import PlanningAgent
from app.agents.inference_agent import InferenceAgent
from app.agents.region_extraction_agent import RegionExtractionAgent
from app.agents.crop_extraction_agent import CropExtractionAgent
from app.agents.region_description_agent import RegionDescriptionAgent
from app.agents.validation_agent import ValidationAgent
from app.agents.analysis_agent import AnalysisAgent
from app.agents.report_agent import ReportAgent
from app.core.logger import logger


class Workflow:
    """
    Full pipeline. ReportAgent is intentionally kept OUTSIDE the main
    loop and always runs at the end -- even if an earlier agent failed
    -- so you always get overlay.png + report.md with whatever partial
    results exist, plus the error list. Silent total failure is worse
    than a partial report.
    """

    def __init__(self):
        self.pipeline = [
            IngestionAgent(),
            PreprocessingAgent(),
            PlanningAgent(),
            InferenceAgent(),
            RegionExtractionAgent(),
            CropExtractionAgent(),
            RegionDescriptionAgent(),
            ValidationAgent(),
            AnalysisAgent(),
        ]
        self.report_agent = ReportAgent()

    def run(self, state):
        for agent in self.pipeline:
            name = agent.__class__.__name__

            try:
                state = agent.run(state)
            except Exception as e:
                logger.error(f"{name} raised an unhandled exception: {e}")
                state.errors.append(f"{name}: {e}")

            if state.errors:
                logger.warning(
                    f"Stopping main pipeline after {name} due to errors: {state.errors}"
                )
                break

        try:
            state = self.report_agent.run(state)
        except Exception as e:
            logger.error(f"ReportAgent raised an unhandled exception: {e}")
            state.errors.append(f"ReportAgent: {e}")

        return state
