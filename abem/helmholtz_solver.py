import numpy as np
from .solver import Solver
from .boundary_solutions import ExteriorBoundarySolution, InteriorBoundarySolution


class HelmholtzSolver(Solver):

    def solve_boundary(self, orientation, k, boundary_condition, boundary_incidence, mu=None):
        mu = mu or (1j / (k + 1))
        assert boundary_condition.f.size == self.len()
        A, B = self.compute_boundary_matrices(k, mu, orientation)
        c = np.empty(self.len(), dtype=complex)
        for i in range(self.len()):
            c[i] = boundary_incidence.phi[i] + mu * boundary_incidence.v[i]
        if 'exterior' == orientation:
            c = -1.0 * c
        else:
            assert 'interior' == orientation, "orientation must be either 'interior' or 'exterior'"

        phi, v = self.solve_linear_equation(B, A, c,
                                            boundary_condition.alpha,
                                            boundary_condition.beta,
                                            boundary_condition.f)
        if 'exterior' == orientation:
            return ExteriorBoundarySolution(self, boundary_condition, k, phi, v)
        else:
            return InteriorBoundarySolution(self, boundary_condition, k, phi, v)
        
    @classmethod
    def solve_linear_equation(cls, Ai, Bi, ci, alpha, beta, f):
        A = np.copy(Ai)
        B = np.copy(Bi)
        c = np.copy(ci)

        x = np.empty(c.size, dtype=np.complex)
        y = np.empty(c.size, dtype=np.complex)

        gamma = np.linalg.norm(B, np.inf) / np.linalg.norm(A, np.inf)
        swapXY = np.empty(c.size, dtype=bool)
        for i in range(c.size):
            if np.abs(beta[i]) < gamma * np.abs(alpha[i]):
                swapXY[i] = False
            else:
                swapXY[i] = True

        for i in range(c.size):
            if swapXY[i]:
                for j in range(alpha.size):
                    c[j] += f[i] * B[j,i] / beta[i]
                    B[j, i] = -alpha[i] * B[j, i] / beta[i]
            else:
                for j in range(alpha.size):
                    c[j] -= f[i] * A[j, i] / alpha[i]
                    A[j, i] = -beta[i] * A[j, i] / alpha[i]

        A -= B
        y = np.linalg.solve(A, c)

        for i in range(c.size):
            if swapXY[i]:
                x[i] = (f[i] - alpha[i] * y[i]) / beta[i]
            else:
                x[i] = (f[i] - beta[i] * y[i]) / alpha[i]

        for i in range(c.size):
            if swapXY[i]:
                temp = x[i]
                x[i] = y[i]
                y[i] = temp

        return x, y

