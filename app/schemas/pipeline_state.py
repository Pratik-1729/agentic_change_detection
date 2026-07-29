from typing import Optional, Dict, Any

from pydantic import BaseModel, Field
from typing import Any

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

    # Preprocessing
    preprocessed_t1: Optional[str] = None
    preprocessed_t2: Optional[str] = None

    # Inference
    probability_map: Optional[Any] = None
    change_mask: Optional[Any] = None

    # Analysis
    statistics: Dict[str, Any] = Field(default_factory=dict)

    # Outputs
    overlay_path: Optional[str] = None
    report_path: Optional[str] = None

    # Errors
    errors: list[str] = Field(default_factory=list)