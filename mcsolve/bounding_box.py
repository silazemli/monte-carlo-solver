from __future__ import annotations
from typing import NamedTuple
import numpy as np
from numpy.typing import NDArray

class BoundingBox(NamedTuple):
    """
    Bounding box in the
        (xmin, ymin, xmax, ymax)
    format.
    """
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def overlaps(self, other: BoundingBox) -> bool:
        return not (self.xmax < other.xmin or
                    self.xmin > other.xmax or
                    self.ymax < other.ymin or
                    self.ymin > other.ymax)

    def inside(self, point: NDArray[np.float64]) -> bool:
        return (self.xmin <= point[0] <= self.xmax and
                self.ymin <= point[1] <= self.ymax)

    @classmethod
    def combine(cls, bounding_boxes: list[BoundingBox]) -> BoundingBox:
        xmin = np.inf
        xmax = -np.inf
        ymin = np.inf
        ymax = -np.inf

        for bounding_box in bounding_boxes:
            xmin = min(xmin, bounding_box.xmin)
            xmax = max(xmax, bounding_box.xmax)
            ymin = min(ymin, bounding_box.ymin)
            ymax = max(ymax, bounding_box.ymax)

        return BoundingBox(xmin, ymin, xmax, ymax)