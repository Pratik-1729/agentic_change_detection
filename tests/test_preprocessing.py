from app.models.changeformer.transforms import (
    load_rgb_tif,
    preprocess
)


img = load_rgb_tif(
    "data/input/t1.tif"
)

print(img.shape)


tensor = preprocess(img)

print(tensor.shape)