from .rayleigh_cavity_solver import RayleighCavitySolver
from .boundary_solutions import SampleSolution


bOptimized = True
if bOptimized:
    from .helmholtz_integrals_3d_c import *
else:
    from .helmholtz_integrals_3d import *


class RayleighCavitySolver3D(RayleighCavitySolver):
    def __init__(self, mesh, c=344.0, density=1.205):
        super(RayleighCavitySolver3D, self).__init__(mesh, c, density)
        self.centers = self.geometry.centers()
        
    def compute_boundary_matrix(self, k, alpha, beta):
        m = self.open_elements
        n = self.total_number_of_elements() - m
        M = np.zeros((2*(m+n), 2*(m+n)), dtype=np.complex64)

        # Compute the top half of the "big matrix".
        for i in range(m+n):
            p = self.centers[i]
            for j in range(m+n):
                qa, qb, qc = self.geometry.triangle_vertices(j)

                element_m = compute_m(k, p, qa, qb, qc, i == j)
                element_l = compute_l(k, p, qa, qb, qc, i == j)

                M[i, j] = -element_m
                M[i, j + m + n] = element_l

            M[i, i] -= 0.5  # subtract half a "identity matrix" from the M-factor submatrix

        # Fill in the bottom half of the "big matrix".
        M[m+n:2*m+n,               0:m] = np.eye(m, dtype=np.float32)
        M[2*m+n:2*(m+n),         m:m+n] = np.diag(alpha)
        M[m+n:2*m+n,         m+n:2*m+n] = 2.0 * M[0:m, m+n:2*m+n]
        M[2*m+n:2*(m+n), 2*m+n:2*(m+n)] = np.diag(beta)
        return M

    def solve_interior(self, solution, samples):
        phis = np.empty(samples.shape[0], dtype=complex)

        for i in range(samples.shape[0]):
            p = samples[i, :]
            sum = 0.0
            for j in range(solution.phis.size):
                qa, qb, qc = self.geometry.triangle_vertices(j)
                element_l = compute_l(solution.k, p, qa, qb, qc, False)
                element_m = compute_m(solution.k, p, qa, qb, qc, False)
                sum += element_l * solution.velocities[j] - element_m * solution.phis[j]
            phis[i] = sum

        return SampleSolution(solution, phis)

    def solve_exterior(self, solution, samples):
        phis = np.empty(samples.shape[0], dtype=complex)

        for i in range(samples.shape[0]):
            p = samples[i, :]
            sum = 0.0
            for j in range(self.open_elements):
                qa, qb, qc = self.geometry.triangle_vertices(j)
                element_l = compute_l(solution.k, p, qa, qb, qc, False)
                sum += -2.0 * element_l * solution.velocities[j]
            phis[i] = sum

        return SampleSolution(solution, phis)
