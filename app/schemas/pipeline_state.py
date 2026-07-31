from typing import Optional, Dict, Any

from pydantic import BaseModel, Field


class PipelineState(BaseModel):
    """
    Shared state passed between all agents.
    """

    # Job
    job_id: str = ""

    # Inputs
    image_t1: Optional[str] = None
    image_t2: Optional[str] = None

    # Validation
    validation_status: bool = False

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Planning
    selected_model: Optional[str] = None
    selected_vlm: Optional[str] = None

    # Preprocessing
    preprocessed_t1: Optional[Any] = None
    preprocessed_t2: Optional[Any] = None

    # Inference
    probability_map: Optional[Any] = None
    change_mask: Optional[Any] = None

    # Region-level VLM analysis (phase 2)
    regions: list[Any] = Field(default_factory=list)            # ChangeRegion boxes
    crops: list[Any] = Field(default_factory=list)               # before/after crops per region
    descriptions: list[Any] = Field(default_factory=list)        # VLM description per crop
    validated_regions: list[Any] = Field(default_factory=list)   # description + TRUE_CHANGE/FALSE_POSITIVE verdict

    # Analysis
    statistics: Dict[str, Any] = Field(default_factory=dict)

    # Outputs
    overlay_path: Optional[str] = None
    report_path: Optional[str] = None

    # Errors
    errors: list[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
