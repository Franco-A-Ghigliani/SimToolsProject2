import numpy as np
from scipy.optimize import root
from squeezer import Seven_bar_mechanism  # Assuming your class is in this file


def constraint_equations(x):
    """
    Defines the system g(q) = 0 that needs to be solved.
    x = [β, γ, Φ, δ, Ω, ε] are the unknowns.
    """
    beta, gamma, phi, delta, omega, epsilon = x

    # Define the equations from Reference [2], Eq. (7.7)
    g1 = beta + 0.0617138900142764496358948458001
    g2 = gamma - 0.455279819163070380255912382449
    g3 = phi - 0.222668390165885884674473185609
    g4 = delta - 0.487364979543842550225598953530
    g5 = omega + 0.222668390165885884674473185609
    g6 = epsilon - 1.23054744454982119249735015568

    return np.array([g1, g2, g3, g4, g5, g6])


def compute_consistent_initial_values():
    """
    Uses Newton’s method to solve g(q) = 0 for consistent initial values.
    """
    initial_guess = np.array([-0.06, 0.45, 0.22, 0.48, -0.22, 1.23])  # Close to expected values

    solution = root(constraint_equations, initial_guess, method='hybr')

    if solution.success:
        return solution.x
    else:
        raise ValueError("Newton's method failed to converge: " + solution.message)


# Solve for consistent initial values
consistent_values = compute_consistent_initial_values()
print("Computed Initial Values:", consistent_values)

# Initialize the Seven_bar_mechanism with these values
mechanism = Seven_bar_mechanism()
y, yp = mechanism.init_squeezer()

# Replace initial values with the computed ones
y[:7] = np.hstack((consistent_values[0], 0.0, consistent_values[1:]))  # Theta(0) = 0
print("Updated Initial Values in Mechanism:", y[:7])

# Test residuals to verify correctness
t = 0.0
residuals = mechanism.f(t, y, yp)
print("Residuals at t=0:", residuals[:7])  # Should be close to zero if valid
