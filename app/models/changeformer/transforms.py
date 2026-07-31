import numpy as np
import torch
import rasterio


def load_rgb_tif(path):

    with rasterio.open(path) as src:

        image = src.read(
            [1,2,3]
        )

    # CHW -> HWC
    image = np.transpose(
        image,
        (1,2,0)
    )

    return image


def preprocess(image):

    image = image.astype(
        np.float32
    )

    # normalize 0-1
    image /= 255.0

    # HWC -> CHW
    image = np.transpose(
        image,
        (2,0,1)
    )

    tensor = torch.from_numpy(
        image
    )

    # add batch dimension
    tensor = tensor.unsqueeze(0)

    return tensor