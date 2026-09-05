from .domain import Domain
from .equation import Equation
from .boundary_segment import BoundarySegment
from .boundary_condition import Dirichlet, Neumann
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree
from itertools import chain
from dataclasses import dataclass, field

class Assembler:
    @dataclass
    class _PointDataArrays:
        points:             list[NDArray[np.float64]] = field(default_factory=list)
        offsets:            list[int] = field(default_factory=list)
        neighbors:          list[int] = field(default_factory=list)
        probabilities:      list[float] = field(default_factory=list)
        constants:          list[float] = field(default_factory=list)
        norms:              list[float] = field(default_factory=list)
        segments:           list[BoundarySegment] = field(default_factory=list)
        distances:          list[float] = field(default_factory=list)
        boundary_reference: list[NDArray[np.float64]] = field(default_factory=list)

    def __init__(self, domain: Domain, equation: Equation,
                 h: float, max_neighbors: int = 12,
                 distance_coefficient: float = 0.625, epsilon: float = 1e-6):
        self.domain = domain
        self.equation = equation
        self.h = h
        self.max_neighbors = max_neighbors
        self.distance_coefficient = distance_coefficient
        self.epsilon = epsilon

        self.inner = self._PointDataArrays()
        self.mirror = self._PointDataArrays()
        self.ghost = self._PointDataArrays()
        self.boundary = self._PointDataArrays()
        
        self._discretize_domain()
        self._discretize_boundaries()
        self._handle_vertices()
        self._handle_grid_points()
        self._build_kdtree()

        self._find_inner_points()
        self._find_near_boundary_points()

        if self.inner.points:
            self._build_inner_stencils()
        else:
            raise RuntimeError("Malformed domain.")

        if self.mirror.points:
            self._build_mirror_stencils()
            self._build_ghost_stencils()
        
        if self.boundary.points:
            self._build_boundary_stencils()

        self._finalize()

    def _discretize_domain(self):
        h = self.h
        xmin, xmax = self.domain.xmin - h, self.domain.xmax + h
        ymin, ymax = self.domain.ymin - h, self.domain.ymax + h

        self.nx = int(np.floor((xmax - xmin) / h)) + 1
        self.ny = int(np.floor((ymax - ymin) / h)) + 1

        self.grid_points: list[NDArray[np.float64]] = []
        for j in range(self.ny):
            for i in range(self.nx):
                self.grid_points.append(np.array([xmin + i*h, ymin + j*h], dtype=np.float64))

    def _discretize_boundaries(self):
        for boundary in self.domain.boundaries:
            for segment in boundary.segments:
                points = segment.discretize(self.h)
                for point in points:
                    self._handle_boundary_point(point, segment)

    def _handle_grid_points(self):
        self.inner_to_grid: list[int] = []
        self.grid_to_inner = -np.ones(len(self.grid_points), dtype=int)
        for index in range(len(self.grid_points)):
            point = self.grid_points[index]
            on_boundary = False
            for segment in self.domain.nearby_segments(point, self.h):
                if segment.distance(point) < 0.25*self.h:
                    on_boundary = True
                    break
            
            if self.domain.inside(point) and not on_boundary:
                self.inner_to_grid.append(index)
                self.grid_to_inner[index] = len(self.inner.points)
                self.inner.points.append(point)

    def _handle_vertices(self):
        for boundary in self.domain.boundaries:
            segments = boundary.segments
            for index in range(len(segments)):
                self._handle_vertex(segments[index], segments[(index + 1) % len(segments)])

    def _handle_vertex(self, left: BoundarySegment, right: BoundarySegment):
        if isinstance(left.boundary_condition, Dirichlet):
            boundary_point = left.shape.point_at_fraction(0.0)

            self.boundary.points.append(boundary_point)
            self.boundary.segments.append(left)
        elif isinstance(right.boundary_condition, Dirichlet):
            boundary_point = right.shape.point_at_fraction(1.0)

            self.boundary.points.append(boundary_point)
            self.boundary.segments.append(right)
        else:
            left_point = left.point_near_end(self.h)
            right_point = right.point_near_start(self.h)

            left_inward = -left.normal(left_point)
            right_inward = -right.normal(right_point)

            A = np.array([
                [left_inward[0], -right_inward[0]],
                [left_inward[1], -right_inward[1]],
            ], dtype=np.float64)

            if abs(np.linalg.det(A)) < 1e-6:
                return

            b = right_point - left_point

            left_distance, right_distance = np.linalg.solve(A, b)

            if left_distance <= 0 or right_distance <= 0:
                return

            mirror_point = left_point + left_distance*left_inward

            if not self.domain.inside(mirror_point):
                return

            mirror_index = len(self.mirror.points)
            self.mirror.points.append(mirror_point)

            left_ghost = 2.0*left_point - mirror_point
            right_ghost = 2.0*right_point - mirror_point

            self.ghost.points.append(left_ghost)
            self.ghost.segments.append(left)
            self.ghost.distances.append(left_distance)
            self.ghost.neighbors.append(mirror_index)
            self.ghost.boundary_reference.append(mirror_point)

            self.ghost.points.append(right_ghost)
            self.ghost.segments.append(right)
            self.ghost.distances.append(right_distance)
            self.ghost.neighbors.append(mirror_index)
            self.ghost.boundary_reference.append(mirror_point)
        
    def _find_inner_points(self):
        self.inner_neighbors_lists: list[list[int]] = [[] for _ in range(len(self.inner.points))]
        for index in range(len(self.inner.points)):
            grid_index = self.inner_to_grid[index]
            i = grid_index % self.nx
            j = grid_index // self.nx

            west = self.grid_to_inner[(i - 1) + j*self.nx] if i - 1 >= 0 else -1
            east = self.grid_to_inner[(i + 1) + j*self.nx] if i + 1 < self.nx else -1
            south = self.grid_to_inner[i + (j - 1)*self.nx] if j - 1 >= 0 else -1
            north = self.grid_to_inner[i + (j + 1)*self.nx] if j + 1 < self.ny else -1

            if west == -1 or east == -1 or south == -1 or north == -1:
                continue

            self.inner_neighbors_lists[index] = [west, east, south, north]

    def _find_near_boundary_points(self):
        near_boundary_indices = [
            index for index, neighbors in enumerate(self.inner_neighbors_lists) if not neighbors
        ] # inner indices of near boundary points
        
        near_boundary_points = np.array(self.inner.points)[near_boundary_indices]
        _, near_boundary_neighbors_lists = self.tree.query(near_boundary_points, k=self.max_neighbors + 1)
        near_boundary_neighbors_lists = near_boundary_neighbors_lists[:, 1:]

        for near_boundary_index, inner_index in enumerate(near_boundary_indices):
            self.inner_neighbors_lists[inner_index] = near_boundary_neighbors_lists[near_boundary_index]

    def _build_inner_stencils(self):
        self.current_data_index = 0
        for index, point in enumerate(self.inner.points):
            self.inner.offsets.append(self.current_data_index)
            neighbors = self.inner_neighbors_lists[index]

            if len(neighbors) == 4:
                weights, constant = self.equation.five_point_stencil(point, self.h)
            else:
                weights, constant = self.equation.monotone_stencil(
                    point, self.h, self.all_points[neighbors]
                )

            probabilities, norm = self._weights_to_probabilities(weights)

            self.inner.neighbors.extend(neighbors)
            self.inner.probabilities.extend(probabilities)
            self.inner.constants.append(constant)
            self.inner.norms.append(norm)

            self.current_data_index += len(probabilities)

    def _build_mirror_stencils(self):
        _, mirror_neighbors_lists = self.tree.query(
            np.array(self.mirror.points), k=self.max_neighbors + 1
        )
        mirror_neighbors_lists = mirror_neighbors_lists[:, 1:]

        for index, point in enumerate(self.mirror.points):
            self.mirror.offsets.append(self.current_data_index)
            neighbors = mirror_neighbors_lists[index]

            weights, constant = self.equation.monotone_stencil(
                point, self.h, self.all_points[neighbors]
            )

            probabilities, norm = self._weights_to_probabilities(weights)

            self.mirror.neighbors.extend(neighbors)
            self.mirror.probabilities.extend(probabilities)
            self.mirror.constants.append(constant)
            self.mirror.norms.append(norm)

            self.current_data_index += len(probabilities)

    def _build_ghost_stencils(self):
        for index, point in enumerate(self.ghost.points):
            self.ghost.offsets.append(self.current_data_index)

            segment = self.ghost.segments[index]
            distance = self.ghost.distances[index]

            weights, constant = segment.build_stencil(point, distance)

            probabilities, norm = self._weights_to_probabilities(weights)
            
            self.ghost.neighbors[index] += len(self.inner.points)
            self.ghost.probabilities.extend(probabilities)
            self.ghost.constants.append(constant)
            self.ghost.norms.append(norm)

            self.current_data_index += len(probabilities)

    def _build_boundary_stencils(self):
        for index, point in enumerate(self.boundary.points):
            self.boundary.offsets.append(self.current_data_index)
            segment = self.boundary.segments[index]

            _, constant = segment.build_stencil(point)

            self.boundary.constants.append(constant)
            self.boundary.norms.append(0.0)

    def _find_mirror_ghost_pair(self,
                                boundary_point: NDArray[np.float64],
                                boundary_segment: BoundarySegment
        ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        normal = boundary_segment.shape.normal(boundary_point)

        distance = self.distance_coefficient*self.h

        while True:
            mirror_point = boundary_point - normal*distance
            if self.domain.inside(mirror_point):
                break
            distance *= 0.5
        
        ghost_point = boundary_point + normal*distance
        return mirror_point, ghost_point, distance
    
    def _build_kdtree(self):
        self.all_points = np.vstack([
            self.inner.points,
            self.mirror.points if self.mirror.points else np.empty((0, 2), dtype=np.float64),
            self.ghost.points if self.ghost.points else np.empty((0, 2), dtype=np.float64),
            self.boundary.points if self.boundary.points else np.empty((0, 2), dtype=np.float64)]
            ).astype(np.float64)
        self.tree = KDTree(self.all_points)
    
    def _handle_boundary_point(self, point: NDArray[np.float64], segment: BoundarySegment):
        if isinstance(segment.boundary_condition, Dirichlet):
            self.boundary.points.append(point)
            self.boundary.segments.append(segment)
        elif isinstance(segment.boundary_condition, Neumann):
            mirror_point, ghost_point, distance = \
            self._find_mirror_ghost_pair(point, segment)
            
            self.ghost.points.append(ghost_point)
            self.ghost.segments.append(segment)
            self.ghost.distances.append(distance)
            self.ghost.neighbors.append(len(self.mirror.points))
            self.ghost.boundary_reference.append(point)

            self.mirror.points.append(mirror_point)

    def _weights_to_probabilities(self, weights: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        if any([weight < 0 for weight in weights]):
            raise RuntimeError(f"Negative stencil weights encountered during stochastic interpretation")
        weights = np.abs(weights)
        norm = weights.sum()
        probabilities = np.cumsum(weights) / norm
        probabilities[-1] = 1.0
        return probabilities, norm
    
    def _finalize(self):
        self.offsets = np.fromiter(chain(
            self.inner.offsets, self.mirror.offsets,
            self.ghost.offsets, self.boundary.offsets,
            [self.current_data_index]), dtype=np.uint32)
        self.neighbors = np.fromiter(chain(
            self.inner.neighbors, self.mirror.neighbors,
            self.ghost.neighbors, self.boundary.neighbors), dtype=np.uint32)
        self.probabilities = np.fromiter(chain(
            self.inner.probabilities, self.mirror.probabilities,
            self.ghost.probabilities, self.boundary.probabilities), dtype=np.float64)
        self.constants = np.fromiter(chain(
            self.inner.constants, self.mirror.constants,
            self.ghost.constants, self.boundary.constants), dtype=np.float64)
        self.norms = np.fromiter(chain(
            self.inner.norms, self.mirror.norms,
            self.ghost.norms, self.boundary.norms), dtype=np.float64)
        self.number_of_inner_points = len(self.inner.points)

    def _construct_starting_point(self, x: float, y: float) -> tuple[
        NDArray[np.uint32], NDArray[np.float64], float, float]:
        point = np.array([x, y], dtype=np.float64)

        indices = self.tree.query(point, k=self.max_neighbors + 1)[1]

        weights, constant = self.equation.monotone_stencil(
            point, self.h, self.all_points[indices]
        )

        probabilities, norm = self._weights_to_probabilities(weights)

        return (
            indices.astype(np.uint32),
            probabilities.astype(np.float64),
            float(constant),
            float(norm),
        )