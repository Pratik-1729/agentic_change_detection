import cv2
from app.detection.region_extractor import RegionExtractor
mask = cv2.imread("change_mask.png",0)

extractor = RegionExtractor(
    min_area = 300
)

regions = extractor.extract(mask)

print(f"detected {len(regions)} regions\n")
for i,region in enumerate(regions):
    print(
        f"{i}: "
        f"x={region.x}, "
        f"y={region.y}, "
        f"width={region.width}, "
        f"height={region.height}, "
        f"area={region.area}, "
        
    )

image = cv2.imread("data/input/before_1.tif")
vis = extractor.draw_regions(
    image,
    regions,
)

cv2.imwrite("detection.png", vis)
print("saved detection.png")