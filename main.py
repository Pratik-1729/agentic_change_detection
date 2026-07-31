"""
Run the full change detection pipeline on a real image pair.

Examples:
  python main.py --image1 data/before.tif --image2 data/after.tif
  python main.py --image1 before.tif --image2 after.tif --model dummy --vlm dummy
"""
import argparse

from app.graph.workflow import Workflow
from app.schemas.pipeline_state import PipelineState


def main():
    parser = argparse.ArgumentParser(description="Run change detection pipeline")
    parser.add_argument("--image1", required=True, help="Path to 'before' image")
    parser.add_argument("--image2", required=True, help="Path to 'after' image")
    parser.add_argument("--model", default="changeformer", help="Registered model name")
    parser.add_argument("--vlm", default="qwen", help="Registered VLM name")
    args = parser.parse_args()

    state = PipelineState(
        image_t1=args.image1,
        image_t2=args.image2,
        selected_model=args.model,
        selected_vlm=args.vlm,
    )

    result = Workflow().run(state)

    print()
    print("job_id:", result.job_id)
    print("errors:", result.errors)
    print("regions:", len(result.regions))
    print("stats:", result.statistics)
    print("overlay:", result.overlay_path)
    print("report:", result.report_path)
    print("json_report:", result.json_report_path)


if __name__ == "__main__":
    main()
