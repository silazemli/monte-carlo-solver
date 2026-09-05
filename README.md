# Monte Carlo PDE Solver

A Monte Carlo solver for partial differential equations (PDEs) on arbitrary 2D domains with curved boundaries. The solver combines a Python frontend for problem setup with a C backend for efficient random walk simulations.

## Overview

This solver uses a probabilistic approach to solve elliptic PDEs of the form:

$$
a_{xx} \frac{\partial^2 u}{\partial x^2} + 2a_{xy} \frac{\partial^2 u}{\partial x \partial y} 
+ a_{yy} \frac{\partial^2 u}{\partial y^2} + b_x \frac{\partial u}{\partial x} + 
b_y \frac{\partial u}{\partial y} + cu = f
$$

on arbitrary 2D domains with Dirichlet, Neumann, or mixed boundary conditions.

The method transforms the PDE into a stochastic representation using the Feynman-Kac formula, then approximates the solution through Monte Carlo simulation of random walks on a computational grid.

### Key Features

- **Arbitrary Domains**: Support for complex domains with curved boundaries defined by line segments and circular arcs
- **Flexible Boundary Conditions**: Dirichlet and Neumann conditions on any boundary segment
- **Adaptive Stencil Construction**: Automatic generation of monotone finite-difference stencils via convex optimization
- **High Performance**: C-based parallel Monte Carlo kernel with OpenMP support
- **Pointwise Evaluation**: Compute solutions at arbitrary interior points without full grid computation

## Architecture

The solver is organized into several components:

### Core Components

- **Domain & Boundary**: Define the computational domain using `Boundary`, `BoundarySegment`, and `Shape` classes (lines and arcs)
- **Boundary Conditions**: `Dirichlet` and `Neumann` classes with customizable functions
- **Equation**: PDE definition with support for Laplace, Poisson, and convection-diffusion equations
- **Assembler**: Builds the stencil representation for all grid points using convex optimization
- **Solver**: Monte Carlo simulation engine using the assembled stencils

### Implementation Flow

1. **Domain Discretization**: The domain is discretized into a structured grid with ghost points for boundary handling
2. **Stencil Assembly**: For each interior point, monotone finite-difference stencils are constructed using convex optimization (via CVXPY and OSQP)
3. **Boundary Handling**: 
   - Dirichlet points: Fixed values from boundary conditions
   - Neumann points: Mirroring technique creates ghost points for flux conditions
   - Vertex handling: Special treatment for corners with mixed conditions
4. **Monte Carlo Simulation**: Random walks follow stencil probabilities until reaching boundary or maximum steps

## Installation

### Prerequisites

- Python 3.8+
- C compiler with OpenMP support (GCC, MSVC, or Clang)
- Required Python packages:
  - `numpy`
  - `scipy`
  - `cvxpy`
  - `rtree`

### Setup

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Compile the C kernel**
   ```bash
   make
   ```

3. **Verify installation**
    ```python
    from mcsolver import Solver, Domain, Laplace
    # Should import without errors
    ```

### Usage

An end-to-end example is provided in [`demo.ipynb`](demo.ipynb).

## Mathematical Method
The solver constructs a Markov chain approximation of the PDE through monotone finite-difference stencils. The stencil weights are computed via convex optimization to satisfy:

Consistency: Preserve the differential operator in the limit h → 0

Monotonicity: All off-diagonal weights are non-negative

Minimal Variance: Minimize squared weights for optimal Monte Carlo efficiency

The resulting random walk transitions between grid points with probabilities proportional to the stencil weights, accumulating contributions from the source term and boundary conditions.