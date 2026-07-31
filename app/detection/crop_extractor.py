from pathlib import Path
import cv2


class CropExtractor:
    def __init__(self, padding=20):
        self.padding = padding

    def crop(self, before, after, regions):
        crops = []
        H, W = before.shape[:2]

        for idx, region in enumerate(regions):
            x = max(0, region.x - self.padding)
            y = max(0, region.y - self.padding)

            x2 = min(W, region.x + region.width + self.padding)
            # was region.width here too — used vertical extent from the
            # box's width instead of its height, so crops were the
            # wrong shape whenever width != height
            y2 = min(H, region.y + region.height + self.padding)

            before_crop = before[y:y2, x:x2]
            after_crop = after[y:y2, x:x2]

            crops.append({
                "id": idx,
                "bbox": (x, y, x2, y2),
                "before": before_crop,
                "after": after_crop,
            })
        return crops

    def save(self, crops, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for crop in crops:
            cv2.imwrite(
                str(output_dir / f"{crop['id']:03d}_before.png"),
                crop["before"]
            )

            cv2.imwrite(
                # was "_before.png" here too — silently overwrote the
                # before-image file and the after crop was never saved
                str(output_dir / f"{crop['id']:03d}_after.png"),
                crop["after"]
            )
