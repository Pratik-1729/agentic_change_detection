from app.models.qwen.qwen_model import QwenModel

model = QwenModel(device="cpu")
model.load()

result = model.generate(
    image="data/input/before_1.tif",
    prompt="Describe this image."
)

print(result)