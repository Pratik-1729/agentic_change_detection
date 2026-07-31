from PIL import Image

from app.models.qwen.qwen_model import QwenModel
from app.agents.region_description_agent import RegionDescriptionAgent

vlm = QwenModel(device="cpu")

vlm.load()

agent =  RegionDescriptionAgent(vlm)

image = Image.open("detection.png").convert("RGB")

results = agent.run([image])
print(results)