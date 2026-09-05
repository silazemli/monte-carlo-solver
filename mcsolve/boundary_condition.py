from abc import ABC, abstractmethod
from typing import Callable
from numpy.typing import NDArray
import numpy as np

class BoundaryCondition(ABC):
    @abstractmethod
    def build_stencil(
        self, point: NDArray[np.float64],
        distance: float | None = None,
    ) -> tuple[NDArray[np.float64], float]:
        """
        Returns (weights, constant), where for the stencil solved for the point it's being built at
            weights are the coefficients that correspond to neighboring points
            constant is the constant part of the right hand side.
        """
        pass

class Dirichlet(BoundaryCondition):
    """ u = g(x, y) """
    def __init__(self, g: Callable[[float, float], float]):
        self.g = g

    def build_stencil(self, point: NDArray[np.float64], distance: float | None = None
                      ) -> tuple[NDArray[np.float64], float]:
        x, y = point

        weights = np.array([], dtype=np.float64)
        constant = self.g(x, y)

        return weights, constant

class Neumann(BoundaryCondition):
    """ ∂u/∂n = g(x, y) """
    def __init__(self, g: Callable[[float, float], float]):
        self.g = g

    def build_stencil(self, point: NDArray[np.float64], distance: float | None = None
                      ) -> tuple[NDArray[np.float64], float]:
        x, y = point

        weights = np.ones(1, dtype=np.float64)
        constant = 2*distance*self.g(x, y)

        return weights, constant
