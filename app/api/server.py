import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

import app.models  # noqa: F401  (registers all models + VLMs)
from app.graph.workflow import Workflow
from app.schemas.pipeline_state import PipelineState
from app.registry.model_registry import ModelRegistry
from app.registry.vlm_registry import VLMRegistry
from app.config.vlm_config import DEFAULT_VLM
from app.core.logger import logger

DEFAULT_MODEL = "changeformer"
UPLOAD_ROOT = Path("uploads")
UPLOAD_ROOT.mkdir(exist_ok=True)

app = FastAPI(title="Change Detection Pipeline API")

JOBS: dict[str, dict] = {}


def _serialize_state(state: PipelineState) -> dict:
    source = state.validated_regions or state.descriptions
    regions = [
        {
            "id": r.get("id"),
            "bbox": r.get("bbox"),
            "description": r.get("description"),
            "decision": r.get("decision"),
            "reason": r.get("reason"),
            "confidence": r.get("confidence"),
        }
        for r in source
    ]
    return {
        "job_id": state.job_id,
        "model": state.selected_model,
        "vlm": state.selected_vlm,
        "statistics": state.statistics,
        "regions": regions,
        "errors": state.errors,
        "overlay_path": state.overlay_path,
        "report_path": state.report_path,
        "json_report_path": state.json_report_path,
    }


def _run_job(job_id: str, state: PipelineState):
    JOBS[job_id]["status"] = "RUNNING"
    try:
        result = Workflow().run(state)

        has_output = bool(result.regions) or result.change_mask is not None
        if result.errors and has_output:
            status = "DONE_WITH_ERRORS"
        elif result.errors:
            status = "FAILED"
        else:
            status = "DONE"

        JOBS[job_id]["status"] = status
        JOBS[job_id]["result"] = _serialize_state(result)
        JOBS[job_id]["state"] = result

    except Exception as e:
        # belt-and-suspenders -- Workflow already catches per-agent
        # errors, this only fires if something outside that (e.g.
        # Workflow() construction itself) blows up
        logger.error(f"Job {job_id} crashed outside the pipeline: {e}")
        JOBS[job_id]["status"] = "FAILED"
        JOBS[job_id]["result"] = {"error": str(e)}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def list_models():
    return {
        "change_detection_models": ModelRegistry.available_models(),
        "vlms": VLMRegistry.available_vlms(),
        "defaults": {"model": DEFAULT_MODEL, "vlm": DEFAULT_VLM},
    }


@app.post("/jobs")
async def submit_job(
    background_tasks: BackgroundTasks,
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    vlm: str = Form(DEFAULT_VLM),
):
    if model.lower() not in ModelRegistry.available_models():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Available: {ModelRegistry.available_models()}",
        )
    if vlm.lower() not in VLMRegistry.available_vlms():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown VLM '{vlm}'. Available: {VLMRegistry.available_vlms()}",
        )

    job_id = str(uuid.uuid4())
    job_dir = UPLOAD_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    path1 = job_dir / image1.filename
    path2 = job_dir / image2.filename

    with open(path1, "wb") as f:
        shutil.copyfileobj(image1.file, f)
    with open(path2, "wb") as f:
        shutil.copyfileobj(image2.file, f)

    state = PipelineState(
        job_id=job_id,
        image_t1=str(path1),
        image_t2=str(path2),
        selected_model=model,
        selected_vlm=vlm,
    )

    JOBS[job_id] = {"status": "PENDING", "result": None, "state": None}
    background_tasks.add_task(_run_job, job_id, state)

    return {"job_id": job_id, "status": "PENDING"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    return {"job_id": job_id, "status": job["status"], "result": job["result"]}


@app.get("/jobs/{job_id}/report")
def get_report(job_id: str, format: str = "json"):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    state: Optional[PipelineState] = job.get("state")
    if state is None or not state.report_path:
        raise HTTPException(status_code=409, detail="job not finished yet")

    path = (
        state.report_path if format == "md"
        else str(Path(state.report_path).with_suffix(".json"))
    )
    if not Path(path).exists():
        raise HTTPException(status_code=404, detail="report file not found")

    return FileResponse(path)


@app.get("/jobs/{job_id}/overlay")
def get_overlay(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    state: Optional[PipelineState] = job.get("state")
    if state is None or not state.overlay_path or not Path(state.overlay_path).exists():
        raise HTTPException(status_code=404, detail="overlay not found")

    return FileResponse(state.overlay_path)