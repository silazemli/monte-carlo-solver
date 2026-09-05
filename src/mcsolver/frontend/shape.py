from __future__ import annotations
from .bounding_box import BoundingBox
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

tau = 2*np.pi
proximity_coefficient = 0.25 # sort of a magic number but whatever

class Shape(ABC):
    
    @property
    @abstractmethod
    def start(self) -> NDArray[np.float64]:
        pass

    @property
    @abstractmethod
    def end(self) -> NDArray[np.float64]:
        pass

    @property
    @abstractmethod
    def bounding_box(self) -> BoundingBox:
        pass
    
    @property
    @abstractmethod
    def length(self) -> float:
        pass

    @abstractmethod
    def normal(self, point: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Unit outward normal vector at a point on the boundary.
        Assumes the point lies at the boundary.
        """
        pass

    @abstractmethod
    def ray_intersections(self, point: NDArray[np.float64], epsilon: float = 1e-8) -> int:
        """Number of crossings with a horizontal half-line from point to positive infinity."""
        pass

    def discretize(self, h: float, margin_coefficient: float = 0.5) -> NDArray[np.float64]:
        """
        Returns a list of points spaced approximately h apart along the segment.
        In case of segment being too small, returns the midpoint.
        """
        if h <= 0:
            raise ValueError("h should be positive.")

        if self.length < h:
            return np.array([self.point_at_fraction(0.5)], dtype=np.float64)
        
        margin_distance = margin_coefficient*h
        margin_fraction = margin_distance / self.length

        usable_length = self.length - 2.0*margin_distance

        if usable_length <= 0:
            return np.array([self.point_at_fraction(0.5)], dtype=np.float64)
        
        n = int(np.ceil(self.length / h))

        fractions = np.linspace(margin_fraction, 1.0 - margin_fraction, n)

        return np.array(
            [self.point_at_fraction(fraction) for fraction in fractions],
            dtype=np.float64
        )
    
    def point_near_start(self, h: float) -> NDArray[np.float64]:
        proximity_fraction = proximity_coefficient*h / self.length
        return self.point_at_fraction(min(0.49, proximity_fraction))
    
    def point_near_end(self, h: float) -> NDArray[np.float64]:
        proximity_fraction = 1.0 - proximity_coefficient*h / self.length
        return self.point_at_fraction(max(0.51, proximity_fraction))
    
    def point_from_start(self, distance: float):
        return self.point_at_fraction(distance / self.length)

    def point_from_end(self, distance: float):
        return self.point_at_fraction(1.0 - distance / self.length)

    @abstractmethod
    def point_at_fraction(self, fraction: float) -> NDArray[np.float64]:
        """
        Returns a point on the segment fragment of the length away from the start.
        """
        pass

    @abstractmethod
    def contains(self, point: NDArray[np.float64], tolerance: float = 1e-6) -> bool:
        """
        Checks if the point lies on the segment.
        """
        pass
    
    @abstractmethod
    def distance(self, point: NDArray[np.float64], epsilon: float = 1e-6) -> float:
        """
        Euclidian distance from point to the closest point on the segment.
        """
        pass

class Line(Shape):
    """
    Directed line segment.
    Interior is to the right of traversal direction.
    """
    def __init__(self, start: NDArray[np.float64], end: NDArray[np.float64], tolerance: float = 1e-6):
        self._start = start
        self._end = end

        self._direction = end - start
        self._length = np.linalg.norm(self._direction)
        self._length_squared = self._length*self._length

        if abs(self._length) < tolerance:
            raise ValueError("Line segment cannot have zero length.")
        
        self._outward_normal = np.array(
            [-self._direction[1], self._direction[0]],
            dtype=np.float64) / self._length

        self._bounding_box = BoundingBox(
            min(self._start[0], self._end[0]), 
            min(self._start[1], self._end[1]), 
            max(self._start[0], self._end[0]), 
            max(self._start[1], self._end[1])
        )

    @property
    def start(self) -> NDArray[np.float64]:
        return self._start
    
    @property
    def end(self) -> NDArray[np.float64]:
        return self._end
    
    @property
    def bounding_box(self) -> BoundingBox:
        return self._bounding_box    

    @property
    def length(self) -> float:
        return self._length
    
    def normal(self, point: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Unit outward normal vector.
        Assumes point is on the infinite supporting line.
        """
        return self._outward_normal
    
    def point_at_fraction(self, fraction: float) -> NDArray[np.float64]:
        return self._start + fraction*self._direction
        
    def contains(self, point: NDArray[np.float64], tolerance: float = 1e-6) -> bool:
        """
        Checks if the point lies on the segment.
        """
        if not self._bounding_box.inside(point):
            return False
        
        vector = point - self._start

        if abs(np.cross(vector, self._direction) / self._length) > tolerance:
            return False
        
        dot = np.dot(vector, self._direction)
        
        return -tolerance <= dot <= self._length_squared + tolerance 

    def distance(self, point: NDArray[np.float64], epsilon: float = 1e-6) -> float:
        """
        Minimum distance from point to the line segment.
        """
        point_to_start = point - self._start

        projection = np.dot(point_to_start, self._direction) / self._length_squared

        projection = max(0.0, min(1.0, projection))

        closest = self._start + projection*self._direction
        return np.linalg.norm(point - closest)
    
    def ray_intersections(self, point: NDArray[np.float64], epsilon: float = 1e-12) -> int:
        px, py = point
        x1, y1 = self._start
        x2, y2 = self._end

        if (y1 > py) == (y2 > py):
            return 0

        t = (py - y1) / (y2 - y1)
        x = x1 + t * (x2 - x1)

        return int(x > px + epsilon)

class Arc(Shape):
    """
    Directed circular arc from start to end.
    Interior is to the right of traversal direction.
    """
    def __init__(self,
                 center: NDArray[np.float64],
                 start: NDArray[np.float64],
                 end: NDArray[np.float64],
                 clockwise: bool = True,
                 tolerance: float = 1e-12):
        self._center = center
        self._start = start
        self._end = end
        self._clockwise = clockwise

        start_vector = start - center
        self._radius = np.linalg.norm(start_vector)
        self._radius_squared = self._radius*self._radius

        if abs(self._radius) < tolerance:
            raise ValueError("Arc radius cannot be zero.")
        
        self._theta_start = np.atan2(start_vector[1], start_vector[0])

        end_vector = end - center
        self._theta_end = np.atan2(end_vector[1], end_vector[0])
        
        if np.allclose(start, end, atol=tolerance):
            self._span = tau
            self._is_full_circle = True
        else:
            if clockwise:
                self._span = (self._theta_start - self._theta_end) % tau
            else:
                self._span = (self._theta_end - self._theta_start) % tau

            self._is_full_circle = np.isclose(self._span, 0.0, atol=tolerance) or \
                                   np.isclose(self._span, tau, atol=tolerance)

        self._length = self._radius*self._span

        if self._is_full_circle:
            self._bounding_box = BoundingBox(
                center[0] - self._radius,
                center[1] - self._radius,
                center[0] + self._radius,
                center[1] + self._radius
            )
            return

        points = [self._start, self._end]

        for theta in [0.0, 0.5*np.pi, np.pi, 1.5*np.pi]:
            candidate = np.array(
                [self._center[0] + self._radius*np.cos(theta),
                 self._center[1] + self._radius*np.sin(theta)],
                dtype=np.float64
            )

            if self._is_point_on_arc(candidate):
                points.append(candidate)
        
        xs: list[float] = [point[0] for point in points]
        ys: list[float] = [point[1] for point in points]

        self._bounding_box = BoundingBox(min(xs), min(ys), max(xs), max(ys))

    @property
    def start(self) -> NDArray[np.float64]:
        return self._start
    
    @property
    def end(self) -> NDArray[np.float64]:
        return self._end
    
    @property
    def bounding_box(self) -> BoundingBox:
        return self._bounding_box

    @property
    def length(self) -> float:
        return self._length
        
    def normal(self, point: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Outward unit normal at boundary point.
        Assumes point lies on the supporting circle.
        """
        radius_vector = point - self._center
        length = np.linalg.norm(radius_vector)
        if length == 0:
            raise ValueError("Normal undefined at center.")

        radial = radius_vector / length
        return radial if self._clockwise else -radial

    def _is_point_on_arc(self, point: NDArray[np.float64], tolerance: float = 1e-6) -> bool:
        if self._is_full_circle:
            return True
        
        point_to_center = point - self._center
        theta: float = np.atan2(point_to_center[1], point_to_center[0]) % tau
        
        if self._clockwise:           
            alpha = (self._theta_start - theta) % tau
            return -tolerance <= alpha <= self._span + tolerance
        else:
            alpha = (theta - self._theta_start) % tau
            return -tolerance <= alpha <= self._span + tolerance
        
    def point_at_fraction(self, fraction: float) -> NDArray[np.float64]:
        if self._clockwise:
            theta = (self._theta_start - fraction*self._span) % tau
        else:
            theta = (self._theta_start + fraction*self._span) % tau

        return np.array([
            self._center[0] + self._radius*np.cos(theta),
            self._center[1] + self._radius*np.sin(theta)],
            dtype=np.float64
        )
    
    def contains(self, point: NDArray[np.float64], tolerance: float = 1e-6) -> bool:
        """
        Checks if the point lies on the segment.
        """
        if not self.bounding_box.inside(point):
            return False
        
        vector = point - self._center
        distance = np.linalg.norm(vector)

        if abs(distance - self._radius) > tolerance:
            return False
        
        if self._is_full_circle:
            return True
        
        theta = np.atan2(vector[1], vector[0])

        if self._clockwise:
            delta = (self._theta_start - theta) % tau
        else:
            delta = (theta - self._theta_start) % tau

        return delta <= self._span + tolerance
    
    def distance(self, point: NDArray[np.float64], epsilon: float = 1e-6) -> float:
        """
        Minimum distance from point to the arc.
        """
        point_to_center = point - self._center
        distance_to_center = np.linalg.norm(point_to_center)

        if abs(distance_to_center) <= epsilon:
            return self._radius

        projected = self._center + (point_to_center / distance_to_center)*self._radius

        if self._is_point_on_arc(projected):
            return abs(distance_to_center - self._radius)

        distance_to_start = np.linalg.norm(point - self._start)
        distance_to_end = np.linalg.norm(point - self._end)

        return min(distance_to_start, distance_to_end)
    
    def ray_intersections(self, point: NDArray[np.float64], epsilon: float = 1e-9) -> int:
        px, py = point

        cy = self._center[1]

        dy = py - cy

        if abs(dy) > self._radius - epsilon:
            return 0

        radicand = self._radius_squared - dy*dy

        if radicand <= epsilon:
            return 0

        dx = np.sqrt(radicand)

        count = 0

        for x in [self._center[0] - dx,
                  self._center[0] + dx]:
            if x <= px + epsilon:
                continue

            candidate = np.array([x, py], dtype=np.float64)

            if not self._is_point_on_arc_half_open(candidate, epsilon):
                continue

            count += 1

        return count
    
    def _is_point_on_arc_half_open(self, point: NDArray[np.float64], epsilon: float = 1e-9) -> bool:
        if self._is_full_circle:
            return True

        if np.linalg.norm(point - self._end) <= epsilon:
            return False

        if np.linalg.norm(point - self._start) <= epsilon:
            return True

        vector = point - self._center

        theta = np.atan2(vector[1], vector[0]) % tau

        if self._clockwise:
            alpha = (self._theta_start - theta) % tau
        else:
            alpha = (theta - self._theta_start) % tau

        return (
            alpha >= -epsilon and
            alpha < self._span - epsilon
        )