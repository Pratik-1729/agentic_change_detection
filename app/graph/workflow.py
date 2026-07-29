from app.agents.ingestion_agent import IngestionAgent
from app.agents.validation_agent import ValidationAgent
from app.agents.preprocessing_agent import PreprocessingAgent
from app.agents.planning_agent import PlanningAgent
from app.agents.inference_agent import InferenceAgent
from app.agents.analysis_agent import AnalysisAgent
from app.agents.report_agent import ReportAgent


class Workflow:

    def __init__(self):

        self.pipeline = [
            IngestionAgent(),
            ValidationAgent(),
            PreprocessingAgent(),
            PlanningAgent(),
            InferenceAgent(),
            AnalysisAgent(),
            ReportAgent(),
        ]

    def run(self, state):

        for agent in self.pipeline:

            state = agent.run(state)

            if state.errors:
                break

        return state