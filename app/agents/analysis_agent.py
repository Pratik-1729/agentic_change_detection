from app.interfaces.base_agent import BaseAgent


class AnalysisAgent(BaseAgent):

    def run(self, state):
        return state