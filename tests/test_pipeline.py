from app.graph.workflow import Workflow
from app.schemas.pipeline_state import PipelineState

# Import to trigger model registration
import app.models.dummy

state = PipelineState(
    image_t1="data/input/t1.png",
    image_t2="data/input/t2.png",
)

workflow = Workflow()

result = workflow.run(state)

print(result.model_dump())