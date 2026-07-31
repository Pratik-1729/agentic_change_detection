import rasterio
import numpy as np
import torch
import cv2


def load_tiff(path):

    with rasterio.open(path) as src:

        img = src.read()

    # CHW -> HWC
    img = np.transpose(
        img,
        (1,2,0)
    )


    # keep first 3 bands
    img = img[:,:,:3]


    # normalize
    img = img.astype(
        np.float32
    )


    img = img / 255.0


    img = cv2.resize(
        img,
        (512,512)
    )


    # HWC -> CHW
    img = np.transpose(
        img,
        (2,0,1)
    )


    tensor = torch.from_numpy(
        img
    )


    tensor = tensor.unsqueeze(0)


    return tensor