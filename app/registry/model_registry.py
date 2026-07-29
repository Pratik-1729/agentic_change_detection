from typing import Dict, Type

from app.interfaces.base_model import BaseChangeDetectionModel


class ModelRegistry:

    _registry: Dict[str, Type[BaseChangeDetectionModel]] = {}


    @classmethod
    def register(
        cls,
        name: str,
        model_class: Type[BaseChangeDetectionModel]
    ):

        key = name.lower()

        if key in cls._registry:
            raise ValueError(
                f"Model already registered: {name}"
            )

        cls._registry[key] = model_class


    @classmethod
    def get(
        cls,
        name: str
    ) -> Type[BaseChangeDetectionModel]:

        key = name.lower()

        if key not in cls._registry:
            raise ValueError(
                f"Unknown model: {name}"
            )

        return cls._registry[key]


    @classmethod
    def available_models(cls):

        return list(cls._registry.keys())


    @classmethod
    def clear(cls):

        cls._registry.clear()