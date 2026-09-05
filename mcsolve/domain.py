from .boundary import Boundary
from .boundary_segment import BoundarySegment
from .bounding_box import BoundingBox
import numpy as np
from numpy.typing import NDArray
from rtree.index import Index

class Domain:
    """
    A domain defined by an arbitrary amount of closed-loop boundaries.
    """
    def __init__(self,
                 xmin: float, xmax: float,
                 ymin: float, ymax: float,
                 boundaries: list[Boundary]
                 ):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.boundaries = boundaries

        self._bounding_box = BoundingBox.combine([boundary.bounding_box for boundary in boundaries])

        self._segments = [segment for boundary in self.boundaries for segment in boundary.segments]

        self._index = Index()
        for index, segment in enumerate(self._segments):
            bounding_box = segment.bounding_box
            self._index.insert(index, bounding_box)

    def inside(self, point: NDArray[np.float64], epsilon: float = 1e-6) -> bool:
        """
        Point-in-domain test via ray casting.
        """
        if not self.bounding_box.inside(point):
            return False
        
        crossings = 0
        for boundary in self.boundaries:
            crossings += boundary.ray_intersections(point, epsilon)
        
        return crossings % 2 == 1

    @property
    def bounding_box(self) -> BoundingBox:
        return self._bounding_box
    
    def nearby_segments(self, point: NDArray[np.float64], margin: float = 1e-3) -> list[BoundarySegment]:
        x, y = point

        query_box = (
            x - margin,
            y - margin,
            x + margin,
            y + margin
        )

        candidate_indices = list(self._index.intersection(query_box))

        return [self._segments[index] for index in candidate_indices]