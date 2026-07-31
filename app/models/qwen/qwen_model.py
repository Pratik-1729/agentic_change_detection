import torch

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
)

from PIL import Image

from app.interfaces.base_vlm import BaseVLM
from app.registry.vlm_registry import VLMRegistry


class QwenModel(BaseVLM):

    def __init__(
        self,
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        device="cpu"
    ):

        self.model_name = model_name
        self.device = device

        self.model = None
        self.processor = None

    def load(self):

        print(f"Loading {self.model_name}...")

        self.processor = AutoProcessor.from_pretrained(
            self.model_name
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            # bfloat16 on CPU (halves memory vs float32, generally fine on
            # modern CPUs); float16 on CUDA. These were previously the
            # same branch by mistake, so device made no difference.
            torch_dtype=torch.bfloat16 if self.device == "cpu" else torch.float16,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)

        self.model.eval()

        print("Qwen loaded successfully.")

    @torch.no_grad()
    def generate(
        self,
        image,
        prompt,
        max_new_tokens=200
    ):

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(self.model.device)
            for k, v in inputs.items()
        }

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

        generated_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], output_ids)
        ]

        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            # False: this is a no-op cleanup step designed for WordPiece
            # tokenizers and is destructive (strips spaces before
            # punctuation) for Qwen's BPE tokenizer — True just emitted
            # a warning and did nothing useful.
            clean_up_tokenization_spaces=False,
        )[0]

        return response.strip()


VLMRegistry.register("qwen", QwenModel)
