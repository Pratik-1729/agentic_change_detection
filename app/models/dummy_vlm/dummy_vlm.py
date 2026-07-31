from app.interfaces.base_vlm import BaseVLM
from app.registry.vlm_registry import VLMRegistry


class DummyVLM(BaseVLM):
    """No weights, no load time. Fixed text output. For pipeline
    plumbing tests only -- not for real descriptions."""

    def load(self):
        pass

    def generate(self, image, prompt, **kwargs):
        if "Decision:" in prompt:
            return "Decision: TRUE_CHANGE\nReason: dummy test\nConfidence: 90"
        return "1. Objects: dummy region.\n2. Land use: unknown.\n3. Signs: none.\n4. Confidence: n/a."


VLMRegistry.register("dummy", DummyVLM)
