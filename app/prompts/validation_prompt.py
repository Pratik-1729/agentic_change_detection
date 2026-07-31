VALIDATION_PROMPT = """
You are an expert in remote sensing and change detection.

You are given:

1. An image crop containing a detected change.
2. A preliminary description of that region.

Description:
{description}

Determine whether this is likely:

- A genuine land-cover or infrastructure change.
- A false positive caused by shadows, illumination, seasonal vegetation, image alignment, or sensor noise.

Return your answer strictly in the following format:

Decision: TRUE_CHANGE or FALSE_POSITIVE

Reason:
<short explanation>

Confidence:
<0-100>
"""