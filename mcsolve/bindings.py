import ctypes
import numpy as np
from numpy.typing import NDArray
import os

_lib_path = os.path.join(os.path.dirname(__file__), "kernel/monte_carlo_solver.dll")
_lib = ctypes.CDLL(_lib_path)

def solve(
        offsets: NDArray[np.uint32],
        neighbors: NDArray[np.uint32],
        probabilities: NDArray[np.float64],
        constants: NDArray[np.float64],
        norms: NDArray[np.float64],
        total_nodes: np.uint32,
        number_of_inner_points: np.uint32,
        max_steps: np.uint32,
        number_of_walks: np.uint32
        ) -> NDArray[np.float64]:
    
    _lib.solve.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_uint32, ctypes.c_uint32,                
        ctypes.c_uint32, ctypes.c_uint32,                
        ctypes.POINTER(ctypes.c_double)
    ]; _lib.solve.restype = None
    
    result = np.zeros(number_of_inner_points, dtype=np.float64)

    _lib.solve(
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        neighbors.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        probabilities.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        constants.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        norms.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        total_nodes, number_of_inner_points, max_steps, number_of_walks,
        result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )

    return result

def solve_at(
        offsets: NDArray[np.uint32],
        neighbors: NDArray[np.uint32],
        probabilities: NDArray[np.float64],
        constants: NDArray[np.float64],
        norms: NDArray[np.float64],

        start_neighbors: NDArray[np.uint32],
        start_probabilities: NDArray[np.float64],
        start_constant: float,
        start_norm: float,

        max_steps: np.uint32,
        number_of_walks: np.uint32
        ) -> float:
    
    _lib.solve_at.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),

        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_uint32,
        ctypes.c_double,
        ctypes.c_double,

        ctypes.c_uint32,
        ctypes.c_uint32,                
    ]; _lib.solve_at.restype = ctypes.c_double
    
    return _lib.solve_at(
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        neighbors.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        probabilities.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        constants.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        norms.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),

        start_neighbors.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        start_probabilities.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(start_neighbors),
        start_constant,
        start_norm,

        max_steps,
        number_of_walks,
    )
