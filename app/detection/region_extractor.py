from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class ChangeRegion:
    x: int
    y:int
    width:int
    height:int
    area:int

    @property
    def bbox(self):
        return(
            self.x,
            self.y,
            self.width,
            self.height,
        )
class RegionExtractor:
    def __init__(
            self,
            min_area:int = 300,
            connectivity: int = 8,
    ):
        self.min_area = min_area
        self.connectivity = connectivity

    def extract(
            self,
            mask: np.ndarray,
    ):
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        if mask.max() == 1:
            mask = mask * 255

        num_labels, labels,stats, centroids = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=self.connectivity,
            )
        ) 

        regions = []

        for i in range(1,num_labels):
            x, y, w, h, area = stats[i]
            if area < self.min_area:
                continue

            regions.append(
                ChangeRegion(
                    x=int(x),
                    y=int(y),
                    width=int(w),
                    height=int(h),
                    area=int(area),
                )
            )
        return regions

    def draw_regions(
            self,
            image,
            regions,
            color=(0,255,0),
            thickness=2,
    ):
        output = image.copy()

        for region in  regions:
            cv2.rectangle(
                output,
                (region.x, region.y),
                (
                    region.x + region.width,
                    region.y + region.height,
                ),
                color,
                thickness,
            )
        return output