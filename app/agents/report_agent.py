from app.interfaces.base_agent import BaseAgent


class ReportAgent(BaseAgent):

    def run(self, state):
        return state