from .boundary_condition import BoundaryCondition
from .bounding_box import BoundingBox
from .shape import Shape
import numpy as np
from numpy.typing import NDArray

class BoundarySegment:
    """
    A boundary segment with represented by a directed shape with a boundary condition.
    """
    def __init__(self, shape: Shape, boundary_condition: BoundaryCondition):
        self.shape = shape
        self.boundary_condition = boundary_condition

    def build_stencil(
            self, point: NDArray[np.float64],
            distance: float | None = None
            ) -> tuple[NDArray[np.float64], float]:
        """
        Returns (weights, constant), where for the stencil solved for the point it's being built at
            weights are the coefficients that correspond to neighboring points
            constant is the constant part of the right hand side.
        """
        return self.boundary_condition.build_stencil(point, distance)
    
    def discretize(self, step: float) -> NDArray[np.float64]:
        """
        Returns a list of points spaced approximately step apart along the segment.
        Points are offset by half-step from start and end.
        In case of segment being too small, returns the middle point.
        """
        return self.shape.discretize(step)
    
    def normal(self, point: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Unit outward normal vector.
        Assumes point is on the infinite supporting line.
        """
        return self.shape.normal(point)
    
    def contains(self, point: NDArray[np.float64], epsilon: float = 1e-6) -> bool:
        return self.shape.contains(point, epsilon)
    
    def distance(self, point: NDArray[np.float64], epsilon: float = 1e-6) -> float:
        return self.shape.distance(point, epsilon)

    def ray_intersections(self, point: NDArray[np.float64], epsilon: float = 1e-12) -> int:
        return self.shape.ray_intersections(point, epsilon)
    
    def point_near_start(self, h: float) -> NDArray[np.float64]:
        return self.shape.point_near_start(h)
    
    def point_near_end(self, h: float) -> NDArray[np.float64]:
        return self.shape.point_near_end(h)

    @property
    def bounding_box(self) -> BoundingBox:
        return self.shape.bounding_box