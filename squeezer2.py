import numpy as np
from assimulo.problem import Implicit_Problem


class Seven_bar_mechanism_2(Implicit_Problem):
    def __init__(self):
        super().__init__()
        self.t0 = 0.0  # Initial time
        self.y0, self.yd0 = self.init_squeezer()

    def init_squeezer(self):
        # Initialize state and derivative as given in the problem
        y_1 = np.array([-0.0617138900142764496358948458001, 0., 0.455279819163070380255912382449,
                        0.222668390165885884674473185609, 0.487364979543842550225598953530,
                        -0.222668390165885884674473185609, 1.23054744454982119249735015568])
        lamb = np.array([98.5668703962410896057654982170, -6.12268834425566265503114393122])
        y = np.hstack((y_1, np.zeros(7), lamb, np.zeros(6)))
        yp = np.hstack((np.zeros(7), np.zeros(7), np.zeros(6)))
        return y, yp

    def res(self, t, y, yp):
        beta, theta, gamma, phi, delta, omega, epsilon = y[:7]
        bep, thp, gap, php, dep, omp, epp = y[7:14]
        lamb = y[14:20]

        # Compute constraint Jacobian G(y)
        G = np.zeros((6, 7))
        G[0, 0] = -np.sin(beta)  # Example modification
        G[1, 1] = np.cos(theta)  # Example modification
        # Complete G based on constraint derivatives

        # Compute residuals
        res_1 = yp[:7] - y[7:14]  # Velocity constraint
        res_2 = np.zeros(7)  # Placeholder for dynamics equation
        res_3 = G @ y[7:14]  # Index-2 constraint G(y) * y' = 0

        return np.hstack((res_1, res_2, res_3))
