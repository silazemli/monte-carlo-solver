from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, NamedTuple
import numpy as np
from numpy.typing import NDArray
import cvxpy as cp

class Equation(ABC):
    """
    Base PDE class.
    """
    class _QuadraticParameters(NamedTuple):
        dx: cp.Parameter
        dy: cp.Parameter
        dx2: cp.Parameter
        dxy: cp.Parameter
        dy2: cp.Parameter
        c: cp.Parameter
        bx: cp.Parameter
        by: cp.Parameter
        axx: cp.Parameter
        axy: cp.Parameter
        ayy: cp.Parameter
        weight: cp.Parameter
    
    class _ProblemTemplate(NamedTuple):
        problem: cp.Problem
        w: cp.Variable
        parameters: Equation._QuadraticParameters

    _templates: dict[int, Equation._ProblemTemplate] = {}

    @abstractmethod
    def five_point_stencil(self, point: NDArray[np.float64], h: float) -> tuple[NDArray[np.float64], float]:
        """
        Returns (weights, constant), where for the stencil solved for the point it's being built at
            weights are the coefficients that correspond to neighboring points
            constant is the constant part of the right hand side
        the neighboring points are west, east, south, north.
        """
        pass

    def monotone_stencil(self, 
                     point: NDArray[np.float64], h: float,
                     neighbors: NDArray[np.float64],
                     epsilon: float = 1e-6
                     ) -> tuple[NDArray[np.float64], float] | None:
        a, b, c, f = self._coefficients(point, h)

        weights = self._calculate_weights(point, neighbors, a, b, c, h)
        if weights is None:
            return None
        else:
            weight_center = weights[0]
            
            weights = -weights[1:] / weight_center
            weights[abs(weights) < epsilon] = 0
            constant = f / weight_center
            
            return weights, constant

    @abstractmethod
    def _coefficients(self, point: NDArray[np.float64], h: float) -> tuple[
                        NDArray[np.float64],
                        NDArray[np.float64],
                        float, float]:
        """
        Returns the PDE coefficients for
        aₓₓuₓₓ + 2aₓᵧuₓᵧ + aᵧᵧuᵧᵧ + bₓuₓ + bᵧuᵧ + cu = f.
        """
        pass

    @classmethod
    def _build_template(cls, n: int) -> _ProblemTemplate:
        w = cp.Variable(n)
        dx, dy, dx2, dxy, dy2 = (cp.Parameter(n) for _ in range(5))
        c, bx, by, axx, axy, ayy = (cp.Parameter() for _ in range(6))
        weight = cp.Parameter(n)
        
        objective = cp.Minimize(cp.sum_squares(cp.multiply(w, weight)) + 1e-6*cp.sum_squares(w))

        constraints = [
            cp.sum(w) == c,
            w @ dx == bx,
            w @ dy == by,
            w @ dx2 == axx,
            w @ dxy == axy,
            w @ dy2 == ayy,
            w[1:] >= 0.0,
            w[0] <= 0.0
        ]
        
        problem = cp.Problem(objective, constraints)

        parameters = Equation._QuadraticParameters(
            dx, dy, dx2, dxy, dy2,
            c, bx, by, axx, axy, ayy,
            weight
        )
        
        return cls._ProblemTemplate(problem, w, parameters)

    @classmethod
    def _get_template(cls, n: int) -> _ProblemTemplate:
        if n not in cls._templates:
            cls._templates[n] = cls._build_template(n)
        return cls._templates[n]
       
    @classmethod
    def _calculate_weights(cls,
                        point: NDArray[np.float64],
                        neighbors: NDArray[np.float64],
                        a: NDArray[np.float64],
                        b: NDArray[np.float64],
                        c: float,
                        h: float) -> NDArray[np.float64]:
        x, y = point
        n = len(neighbors)

        problem, v, parameters = cls._get_template(n + 1)

        dx = np.concatenate(([0.0], (neighbors[:,0] - x) / h))
        dy = np.concatenate(([0.0], (neighbors[:,1] - y) / h))

        parameters.dx.value, parameters.dy.value = dx, dy
        parameters.dx2.value, parameters.dxy.value, parameters.dy2.value = (
            dx*dx, dx*dy, dy*dy
        )

        parameters.c.value = c*h*h
        parameters.bx.value = b[0]*h
        parameters.by.value = b[1]*h
        parameters.axx.value = 2.0*a[0, 0]
        parameters.axy.value = 2.0*a[0, 1]
        parameters.ayy.value = 2.0*a[1, 1]

        weight_scaled = np.maximum(dx*dx + dy*dy, np.finfo(float).eps)
        parameters.weight.value = weight_scaled

        try:
            problem.solve(solver=cp.OSQP,)
        except cp.error.SolverError:
            raise RuntimeError("Solver error occurred")
        
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"Non-optimal solution at ({x}, {y}): {problem.status}")

        if problem.status in ["optimal_inaccurate"]:
            print("Inaccurate QP solution")
            
        if v.value is None:
            raise RuntimeError("Solver failure")

        w = v.value / (h*h)
        return w
    
class Laplace(Equation):
    """ Δu = 0 """
    def __init__(self):
        pass

    def five_point_stencil(self, point: NDArray[np.float64], h: float
                           ) -> tuple[NDArray[np.float64], float]:
        weights = np.full(4, 0.25, dtype=np.float64)
        constant = 0.0

        return weights, constant
    
    def _coefficients(self, point: NDArray[np.float64], h: float) -> tuple[
                        NDArray[np.float64],
                        NDArray[np.float64],
                        float, float]:
        a = np.array([[1.0, 0.0],
                      [0.0, 1.0]],
                      dtype=np.float64)
        b = np.zeros(2, dtype=np.float64)
        c = 0.0
        f = 0.0
        
        return a, b, c, f

class Poisson(Equation):
    """ Δu = f(x, y) """
    def __init__(self, f: Callable[[float, float], float]):
        self.f = f
    
    def five_point_stencil(self, point: NDArray[np.float64], h: float
                           ) -> tuple[NDArray[np.float64], float]:
        x, y = point

        weights = np.full(4, 0.25, dtype=np.float64)
        constant = -0.25*h*h*self.f(x, y)
        return weights, constant 
    
    def _coefficients(self, point: NDArray[np.float64], h: float)-> tuple[
                        NDArray[np.float64],
                        NDArray[np.float64],
                        float, float]:
        x, y = point

        a = np.array([[1.0, 0.0],
                      [0.0, 1.0]],
                     dtype=np.float64)
        b = np.zeros(2, dtype=np.float64)
        c = 0.0
        f = self.f(x, y)

        return a, b, c, f
    
class ConvectionDiffusion(Equation):
    """ Δu + b∇u = f(x,y) """
    def __init__(
        self,
        bx: float,
        by: float,
        f: Callable[[float, float], float] = lambda x, y: 0.0,
    ):
        self.bx = bx
        self.by = by
        self.f = f

    def five_point_stencil(
        self, point: NDArray[np.float64], h: float
    ) -> tuple[NDArray[np.float64], float]:
        neighbors = np.array([
            [point[0] - h, point[1]],
            [point[0] + h, point[1]],
            [point[0], point[1] - h],
            [point[0], point[1] + h],
        ], dtype=np.float64)
        return self.monotone_stencil(point, h, neighbors)

    def _coefficients(
        self, point: NDArray[np.float64], h: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
        a = np.eye(2, dtype=np.float64)
        b = np.array([self.bx, self.by], dtype=np.float64)
        c = 0.0
        f_val = self.f(*point)
        return a, b, c, f_val
    
