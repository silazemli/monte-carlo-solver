from .boundary_segment import BoundarySegment
from .bounding_box import BoundingBox
from numpy.typing import NDArray
import numpy as np

class Boundary:
    """
    A closed-loop boundary consisting of boundary segments.
    """
    def __init__(self, segments: list[BoundarySegment], epsilon=1e-6):
        if not self._is_closed(segments, epsilon):
            raise ValueError("Malformed boundary.")
        
        self.segments = segments
        self._bounding_box = BoundingBox.combine([segment.bounding_box for segment in segments])

    def _is_closed(self, segments: list[BoundarySegment], epsilon: float = 1e-6) -> bool:
        if not segments:
            return False
        
        for index in range(len(segments) - 1):
            if not np.allclose(segments[index].shape.end, segments[index + 1].shape.start, atol=epsilon):
                return False
        
        return np.allclose(segments[-1].shape.end, segments[0].shape.start, atol=epsilon)
    
    def ray_intersections(self, point: NDArray[np.float64], epsilon: float = 1e-6) -> int:
        """
        Total number of intersections between a horizontal ray to positive infinity and this boundary.
        """
        crossings = 0

        for segment in self.segments:
            crossings += segment.ray_intersections(point, epsilon)

        return crossings

    @property
    def bounding_box(self) -> BoundingBox:
        return self._bounding_box