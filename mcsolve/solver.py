from .assembler import Assembler
from .domain import Domain
from .equation import Equation
from .bindings import solve, solve_at
import numpy as np
from numpy.typing import NDArray

class Solver:
    """
    Solve the system built by assembler via a random walk.
    """
    def __init__(self, domain: Domain, equation: Equation, h: float, max_neighbors: int = 25):
        self.domain = domain
        self.equation = equation
        self.h = h
        self.max_neighbors = max_neighbors
        self.result = None

        self.assembler = Assembler(self.domain, self.equation, self.h, self.max_neighbors)

    def solve(self, max_steps: int, number_of_walks: int) -> NDArray[np.float64]:
        assembler = self.assembler

        total_nodes = len(assembler.constants)
        
        result = solve(
            assembler.offsets,
            assembler.neighbors,
            assembler.probabilities,
            assembler.constants,
            assembler.norms,
            total_nodes,
            assembler.number_of_inner_points,
            max_steps, number_of_walks
        )

        nx = self.assembler.nx
        ny = self.assembler.ny
        
        grid = np.full((ny, nx), np.nan, dtype=np.float64)
        for inner_index, grid_index in enumerate(assembler.inner_to_grid):
            i = grid_index % nx
            j = grid_index // nx
            grid[j, i] = result[inner_index]
        
        return grid
    
    def solve_at(self, x: float, y: float, max_steps: int, number_of_walks: int) -> NDArray[np.float64]:
        assembler = self.assembler
        if not self.domain.inside(np.array([x, y], dtype=float)):
            raise ValueError("Point must be inside the domain")
        start_neighbors, start_probabilities, start_constant, start_norm = (
            assembler._construct_starting_point(x, y)
        )
        
        return solve_at(
            assembler.offsets,
            assembler.neighbors,
            assembler.probabilities,
            assembler.constants,
            assembler.norms,

            start_neighbors,
            start_probabilities,
            start_constant,
            start_norm,

            max_steps,
            number_of_walks,
        )
