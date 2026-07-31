from app.registry.model_registry import ModelRegistry
from app.core.logger import logger


class ModelFactory:

    @staticmethod
    def create(model_name: str, device: str = "cpu", **kwargs):
        """
        kwargs are model-specific constructor args (e.g. checkpoint=...)
        sourced from app/config/model_config.py — this stays generic
        so no per-model branching is needed here.
        """
        model_cls = ModelRegistry.get(model_name)

        logger.info(f"Creating model '{model_name}' on device={device}")

        model = model_cls(device=device, **kwargs)
        model.load()

        return model
