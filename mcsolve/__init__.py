from .domain import Domain
from .boundary import Boundary
from .boundary_segment import BoundarySegment
from .shape import Line, Arc
from .solver import Solver
from .equation import Laplace, Poisson, ConvectionDiffusion
from .boundary_condition import Dirichlet, Neumann

__all__ = [
    "Solver",
    # Geometry
    "Domain",
    "Boundary",
    "BoundarySegment",
    "Line",
    "Arc",
    # Equations
    "Laplace",
    "Poisson",
    "ConvectionDiffusion",
    # Boundary conditions
    "Dirichlet",
    "Neumann",
]