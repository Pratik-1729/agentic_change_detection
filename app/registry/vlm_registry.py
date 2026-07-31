from typing import Dict, Type

from app.interfaces.base_vlm import BaseVLM


class VLMRegistry:

    _registry: Dict[str, Type[BaseVLM]] = {}

    @classmethod
    def register(cls, name: str, vlm_class: Type[BaseVLM]):
        key = name.lower()

        if key in cls._registry:
            raise ValueError(f"VLM already registered: {name}")

        cls._registry[key] = vlm_class

    @classmethod
    def get(cls, name: str) -> Type[BaseVLM]:
        key = name.lower()

        if key not in cls._registry:
            raise ValueError(f"Unknown VLM: {name}")

        return cls._registry[key]

    @classmethod
    def available_vlms(cls):
        return list(cls._registry.keys())

    @classmethod
    def clear(cls):
        cls._registry.clear()
