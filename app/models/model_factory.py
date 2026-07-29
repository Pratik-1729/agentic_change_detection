from app.registry.model_registry import ModelRegistry


class ModelFactory:

    @staticmethod
    def create(model_name: str, device: str = "cpu"):

        model_cls = ModelRegistry.get(model_name)

        model = model_cls(device=device)

        model.load()

        return model