import torch
import rasterio
import numpy as np
import cv2

from app.models.changeformer.changeformer_model import ChangeFormerModel


def load_tiff(path):

    with rasterio.open(path) as src:
        img = src.read()

    # CHW -> HWC
    img = np.transpose(img, (1,2,0))

    # RGB only
    img = img[:,:,:3]

    img = img.astype(np.float32)

    # normalize
    img = img / 255.0

    # HWC -> CHW
    img = np.transpose(img,(2,0,1))

    tensor = torch.from_numpy(img)

    return tensor.unsqueeze(0)



model = ChangeFormerModel(
    device="cpu",
    checkpoint="weights/changeformer/best_ckpt.pt"
)

model.load()


before = load_tiff(
    "data/input/before_1.tif"
)

after = load_tiff(
    "data/input/after_1.tif"
)


with torch.no_grad():

    output = model.predict(
        before,
        after
    )


print("Output:", output.shape)


prob = torch.softmax(
    output,
    dim=1
)


mask = torch.argmax(
    prob,
    dim=1
)


mask = mask.squeeze().numpy()


print(
    "Changed pixels:",
    np.sum(mask==1)
)


cv2.imwrite(
    "change_mask.png",
    mask*255
)


print("Saved change_mask.png")