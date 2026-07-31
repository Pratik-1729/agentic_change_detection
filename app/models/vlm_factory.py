from app.registry.vlm_registry import VLMRegistry
from app.core.logger import logger


class VLMFactory:
    """
    Unlike ModelFactory, this caches loaded instances by
    (vlm_name, kwargs). VLMs are expensive to load (large weights,
    slow on CPU) and multiple agents in the same pipeline run
    (RegionDescriptionAgent, ValidationAgent) need the same one —
    without caching, each agent's own VLMFactory.create() call would
    reload the full model from scratch.
    """

    _cache = {}

    @classmethod
    def create(cls, vlm_name: str, **kwargs):
        key = (vlm_name.lower(), tuple(sorted(kwargs.items())))

        if key not in cls._cache:
            vlm_cls = VLMRegistry.get(vlm_name)

            logger.info(f"Loading VLM '{vlm_name}' (kwargs={kwargs})")

            instance = vlm_cls(**kwargs)
            instance.load()

            cls._cache[key] = instance

        return cls._cache[key]

    @classmethod
    def clear_cache(cls):
        cls._cache.clear()
