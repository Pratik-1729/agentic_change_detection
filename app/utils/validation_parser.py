import re

DECISION_RE = re.compile(r"Decision:\s*(TRUE_CHANGE|FALSE_POSITIVE)", re.IGNORECASE)
REASON_RE = re.compile(r"Reason:\s*(.*?)\s*(?=Confidence:|$)", re.IGNORECASE | re.DOTALL)
CONFIDENCE_RE = re.compile(r"Confidence:\s*(\d+)", re.IGNORECASE)


def parse_validation_result(text: str) -> dict:
    """
    Parses the VLM's free-text validation output (see
    app/prompts/validation_prompt.py for the expected format) into
    structured fields. Falls back to "UNKNOWN"/None if the model
    didn't follow the format -- never raises, since a malformed VLM
    reply shouldn't crash the pipeline.
    """
    decision_match = DECISION_RE.search(text or "")
    reason_match = REASON_RE.search(text or "")
    confidence_match = CONFIDENCE_RE.search(text or "")

    return {
        "decision": decision_match.group(1).upper() if decision_match else "UNKNOWN",
        "reason": reason_match.group(1).strip() if reason_match else None,
        "confidence": int(confidence_match.group(1)) if confidence_match else None,
    }
