from pathlib import Path
import json
import cv2
import numpy as np

from app.interfaces.base_agent import BaseAgent
from app.detection.region_extractor import RegionExtractor
from app.core.logger import logger

# NOTE: adjust to wherever you want job outputs written, or wire this
# up to config/env instead of a hardcoded relative path.
OUTPUT_ROOT = Path("outputs")


class ReportAgent(BaseAgent):
    """
    Writes the pipeline's results to disk:
      - overlay.png: the "after" image with detected regions boxed
      - report.md: a markdown summary of statistics + per-region
        descriptions/validations

    Both are written under outputs/<job_id>/.
    """

    def __init__(self, extractor=None):
        self.extractor = extractor or RegionExtractor()

    def run(self, state):
        try:
            job_dir = OUTPUT_ROOT / (state.job_id or "unknown_job")
            job_dir.mkdir(parents=True, exist_ok=True)

            state.overlay_path = self._save_overlay(state, job_dir)
            state.report_path = self._save_report(state, job_dir)
            state.json_report_path =  self._save_json_report(state, job_dir)

            logger.info(f"Report written to {job_dir}")

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            state.errors.append(f"ReportAgent: {e}")

        return state

    def _save_overlay(self, state, job_dir):
        if state.preprocessed_t2 is None:
            return None

        image = np.asarray(state.preprocessed_t2)

        # images are loaded as RGB (see preprocessing_agent.py); cv2
        # reads/writes assuming BGR, so convert before drawing/saving
        # or the overlay's colors (including the green boxes) come out
        # channel-swapped.
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if state.regions:
            image_bgr = self.extractor.draw_regions(image_bgr, state.regions)

        overlay_path = job_dir / "overlay.png"
        cv2.imwrite(str(overlay_path), image_bgr)

        return str(overlay_path)

    def _save_report(self, state, job_dir):
        lines = [
            "# Change Detection Report",
            "",
            f"**Job ID:** {state.job_id}",
            f"**Model:** {state.selected_model}",
            f"**VLM:** {state.selected_vlm or 'n/a'}",
            "",
            "## Summary",
            "",
        ]

        for key, value in (state.statistics or {}).items():
            lines.append(f"- **{key}**: {value}")

        lines.append("")
        lines.append("## Regions")
        lines.append("")

        # prefer validated regions (has the TRUE_CHANGE/FALSE_POSITIVE
        # verdict); fall back to plain descriptions if validation
        # didn't run or produced nothing
        source = state.validated_regions or state.descriptions

        if not source:
            lines.append("No regions were flagged for review.")
        else:
            for region in source:
                lines.append(f"### Region {region['id']}")
                lines.append(f"- bbox: {region.get('bbox')}")
                lines.append(f"- description: {region.get('description', 'n/a')}")
                if "decision" in region:
                    lines.append(f"- decision: {region['decision']}")
                    lines.append(f"- reason: {region.get('reason', 'n/a')}")
                    lines.append(f"- confidence: {region.get('confidence', 'n/a')}")
                lines.append("")

        if state.errors:
            lines.append("## Errors")
            lines.append("")
            for err in state.errors:
                lines.append(f"- {err}")

        report_path = job_dir / "report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")

        return str(report_path)

    def _save_json_report(self, state, job_dir):
            source = state.validated_regions or state.descriptions
    
            regions_json = [
                {
                    "id": r.get("id"),
                    "bbox": r.get("bbox"),
                    "description": r.get("description"),
                    "decision": r.get("decision"),
                    "reason": r.get("reason"),
                    "confidence": r.get("confidence"),
                }
                # drop "crop" (PIL.Image) -- not JSON serializable and not
                # useful in a JSON report anyway; overlay.png already
                # shows the boxes visually
                for r in source
            ]
    
            payload = {
                "job_id": state.job_id,
                "model": state.selected_model,
                "vlm": state.selected_vlm,
                "statistics": state.statistics,
                "regions": regions_json,
                "errors": state.errors,
                "metadata": state.metadata,
                "overlay_path": state.overlay_path,
            }
    
            json_path = job_dir / "report.json"
            json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    
            return str(json_path)