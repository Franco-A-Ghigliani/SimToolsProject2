# -*- coding: utf-8 -*-
from assimulo.solvers import IDA
from assimulo.problem import Implicit_Problem
from squeezer import Seven_bar_mechanism
from squeezer2 import Seven_bar_mechanism_2
import numpy as np


# Define the simulation function
def simulate_squeezer(problem, tf=0.03, algvars=None, atol=None):
    """
    Simulate the squeezer mechanism using IDA.

    Parameters:
        problem: The problem instance (Seven_bar_mechanism or Seven_bar_mechanism_2).
        tf: Final simulation time.
        algvars: Indices of algebraic variables.
        atol: Absolute tolerance vector.
    """
    # Create the IDA solver
    solver = IDA(problem)

    # Set tolerances
    if atol is not None:
        solver.atol = atol
    solver.rtol = 1e-6  # Relative tolerance
    if algvars is not None:
        solver.algvar = algvars

    # Simulate
    t, y, yd = solver.simulate(tf)

    return t, y, yd


# Define the main function
def main():
    # Problem 1: Seven_bar_mechanism
    problem1 = Seven_bar_mechanism()

    # Define algebraic variables for problem1 (λ and velocity components)
    algvars1 = [0] * 20  # Initialize all variables as differential
    algvars1[14:20] = [1] * 6  # Set λ as algebraic variables
    algvars1[7:14] = [1] * 7  # Set velocity components as algebraic variables

    # Define absolute tolerance vector for problem1
    atol1 = [1e-6] * 20  # Default tolerance
    atol1[14:20] = [1e5] * 6  # Large tolerance for λ
    atol1[7:14] = [1e5] * 7  # Large tolerance for velocity components

    # Simulate problem1
    t1, y1, yd1 = simulate_squeezer(problem1, tf=0.03, algvars=algvars1, atol=atol1)
    print("Simulation of Seven_bar_mechanism completed.")

    # Problem 2: Seven_bar_mechanism_2
    problem2 = Seven_bar_mechanism_2()

    # Define algebraic variables for problem2 (λ and velocity components)
    algvars2 = [0] * 20  # Initialize all variables as differential
    algvars2[14:20] = [1] * 6  # Set λ as algebraic variables
    algvars2[7:14] = [1] * 7  # Set velocity components as algebraic variables

    # Define absolute tolerance vector for problem2
    atol2 = [1e-6] * 20  # Default tolerance
    atol2[14:20] = [1e5] * 6  # Large tolerance for λ
    atol2[7:14] = [1e5] * 7  # Large tolerance for velocity components

    # Simulate problem2
    t2, y2, yd2 = simulate_squeezer(problem2, tf=0.03, algvars=algvars2, atol=atol2)
    print("Simulation of Seven_bar_mechanism_2 completed.")


# Run the main function
if __name__ == "__main__":
    main()