from PIL import Image

from app.models.qwen.qwen_model import QwenModel
from app.agents.region_description_agent import RegionDescriptionAgent
from app.agents.validation_agent import ValidationAgent

vlm = QwenModel(device="cpu")
vlm.load()

description_agent = RegionDescriptionAgent(vlm)
validation_agent = ValidationAgent(vlm)

image = Image.open("detection.png").convert("RGB")

regions = description_agent.run([image])

validated = validation_agent.run(regions)

print(validated[0]["validation"])
