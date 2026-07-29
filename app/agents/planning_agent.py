from app.agents.base_agent import BaseAgent


class PlanningAgent(BaseAgent):

    def run(self, state):
        state.selected_model = "dummy"
        return state