"""
Fast plumbing test. Runs full Workflow with DummyModel + DummyVLM --
no checkpoint, no GPU, no slow generation. Confirms wiring works
before you spend time on real ChangeFormer + Qwen runs.
"""

from app.graph.workflow import Workflow
from app.schemas.pipeline_state import PipelineState

state = PipelineState(
    image_t1="data/input/before_1.tif",
    image_t2="data/input/before_1.tif",  # swap for a real 'after' image
    selected_model="dummy",
    selected_vlm="dummy",
)

result = Workflow().run(state)

print("errors:", result.errors)
print("regions found:", len(result.regions))
print("crops:", len(result.crops))
print("descriptions:", len(result.descriptions))
print("validated:", len(result.validated_regions))
print("stats:", result.statistics)
print("overlay:", result.overlay_path)
print("report:", result.report_path)
