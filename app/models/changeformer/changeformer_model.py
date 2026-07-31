import torch
import os
import sys

from app.interfaces.base_model import BaseChangeDetectionModel
from app.registry.model_registry import ModelRegistry
from app.models.changeformer.transforms import preprocess as to_tensor


class ChangeFormerModel(BaseChangeDetectionModel):

    def __init__(
        self,
        device="cpu",
        checkpoint=None
    ):
        super().__init__(device)

        self.checkpoint = checkpoint

    def load(self):

        print("Loading ChangeFormerV6...")

        external_path = os.path.join(
            os.path.dirname(__file__),
            "external"
        )

        if external_path not in sys.path:
            sys.path.insert(0, external_path)

        from models.ChangeFormer import ChangeFormerV6

        self.model = ChangeFormerV6(embed_dim=256)

        if self.checkpoint and os.path.exists(self.checkpoint):

            print(f"Loading weights: {self.checkpoint}")

            checkpoint = torch.load(
                self.checkpoint,
                map_location=self.device,
                weights_only=False
            )

            if "model_G_state_dict" in checkpoint:
                print("Detected training checkpoint")
                state_dict = checkpoint["model_G_state_dict"]
            else:
                print("Detected raw state_dict")
                state_dict = checkpoint

            self.model.load_state_dict(state_dict)

            print("Weights loaded successfully")

        else:
            print("WARNING: checkpoint missing. Using random weights.")

        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, image1, image2):
        # was a no-op passthrough -- model needs normalized CHW
        # tensors with a batch dim, not raw HWC uint8 arrays
        img1 = to_tensor(image1).to(self.device)
        img2 = to_tensor(image2).to(self.device)
        return img1, img2

    @torch.no_grad()
    def predict(self, image1, image2):
        outputs = self.model(image1, image2)
        # final full-resolution prediction
        prediction = outputs[-1]
        return prediction

    def postprocess(self, prediction):
        prediction = torch.softmax(prediction, dim=1)
        mask = torch.argmax(prediction, dim=1)  # shape: (B, H, W)
        mask = mask.cpu().numpy()
        # batch size is always 1 here (InferenceAgent runs one pair at
        # a time) -- drop the batch dim so downstream (RegionExtractor,
        # cv2) gets plain 2D (H, W), not (1, H, W)
        return mask[0]


ModelRegistry.register("changeformer", ChangeFormerModel)