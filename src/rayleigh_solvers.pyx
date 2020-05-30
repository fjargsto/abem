from solver import Solver
from acoustic_boundaries import SampleSolution, RayleighBoundarySolution, RayleighCavityBoundarySolution
from iops_pyx import * # Cython integral operators (slow!)
# from iops_pyx import * # C++ integral operators
import numpy as np


class RayleighSolver(Solver):

    def solve_boundary(self, k, boundary_condition):
        assert boundary_condition.f.size == self.len()
        M = self.compute_boundary_matrix(k, boundary_condition.alpha, boundary_condition.beta)
        n = self.len()
        b = np.zeros(2 * n, dtype=np.complex)
        b[n: 2*n] = boundary_condition.f
        x = np.linalg.solve(M, b)
        
        return RayleighBoundarySolution(self, boundary_condition, k,
                                        x[0:self.len()],
                                        x[self.len():2 * self.len()])

# -----------------------------------------------------------------------------
# Rayleigh solver 3d
# -----------------------------------------------------------------------------
class RayleighSolver3D(RayleighSolver):
    def __init__(self, mesh, c=344.0, density=1.205):
        super(RayleighSolver3D, self).__init__(mesh, c, density)
        self.centers = self.geometry.centers()
        self.areas = None
                                  
    def element_area(self, named_partition=None):
        return self.geometry.areas(named_partition)

    def compute_boundary_matrix(self, k, alpha, beta):
        n = self.len()
        M = np.zeros((2*n, 2*n), dtype=np.complex64)

        # Compute the top half of the "big matrix".
        for i in range(n):
            p = self.centers[i]
            for j in range(n):
                qa, qb, qc = self.geometry.triangle_vertices(j)
                element_l = l_3d(k, p, qa, qb, qc, i == j)
                M[i, j + n] = 2 * element_l

        # Fill in the bottom half of the "big matrix".
        M[0:n,       0:n] = np.eye(n, dtype=np.float32)
        M[n: 2*n,    0:n] = np.diag(alpha)
        M[n: 2*n, n: 2*n] = np.diag(beta)
        
        return M

    def solve_samples(self, solution, samples):
        results = np.empty(samples.shape[0], dtype=complex)

        for i in range(samples.shape[0]):
            p = samples[i]
            sum = 0.0 + 0.0j
            for j in range(solution.phis.size):
                qa, qb, qc = self.geometry.triangle_vertices(j)
                element_l = l_3d(solution.k, p, qa, qb, qc, False)
                sum -= 2.0 * element_l * solution.velocities[j]
            results[i] = sum
        return results

# -----------------------------------------------------------------------------
# Rayleigh cavity solvers
# -----------------------------------------------------------------------------
class RayleighCavitySolver(Solver):
    def __init__(self, geometry, c=344.0, density=1.205):
        super(RayleighCavitySolver, self).__init__(geometry, c, density)
        self.open_elements = self.geometry.named_partition['interface'][1]

    def number_of_interface_elements(self):
        return self.open_elements

    def len(self):
        """The number of elements forming the cavity."""
        return self.total_number_of_elements() - self.open_elements

    def total_number_of_elements(self):
        return super(RayleighCavitySolver, self).len()

    def solve_boundary(self, k, boundary_condition):
        M = self.compute_boundary_matrix(k,
                                         boundary_condition.alpha,
                                         boundary_condition.beta)
        number_of_elements = self.total_number_of_elements()
        b = np.zeros(2*number_of_elements, dtype=np.complex64)
        b[number_of_elements + self.open_elements: 2 * number_of_elements] = boundary_condition.f
        x = np.linalg.solve(M, b)

        return RayleighCavityBoundarySolution(self, boundary_condition, k,
                                              x[0:number_of_elements],
                                              x[number_of_elements:2*number_of_elements])

# -----------------------------------------------------------------------------
# Rayleight cavity solver 3D
# -----------------------------------------------------------------------------
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

                element_m = m_3d(k, p, qa, qb, qc, i == j)
                element_l = l_3d(k, p, qa, qb, qc, i == j)

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
                element_l = l_3d(solution.k, p, qa, qb, qc, False)
                element_m = m_3d(solution.k, p, qa, qb, qc, False)
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
                element_l = l_3d(solution.k, p, qa, qb, qc, False)
                sum += -2.0 * element_l * solution.velocities[j]
            phis[i] = sum

        return SampleSolution(solution, phis)


# -----------------------------------------------------------------------------
# Rayleigh cavity solver RAD
# -----------------------------------------------------------------------------
class RayleighCavitySolverRAD(RayleighCavitySolver):
    def __init__(self, chain, c=344.0, density=1.205):
        super(RayleighCavitySolverRAD, self).__init__(chain, c, density)
        self.centers = self.geometry.centers()
        self.lengths = self.geometry.lengths()
        self.normals = self.geometry.normals()
        self.areas = None

    def element_area(self, named_partition=None):
        return self.geometry.areas(named_partition)

    def cavity_normals(self):
        cavity_start = self.geometry.named_partition['cavity'][0]
        cavity_end = self.geometry.named_partition['cavity'][1]
        return self.normals[cavity_start:cavity_end, :]

    def compute_boundary_matrix(self, k, alpha, beta):
        m = self.open_elements
        n = self.total_number_of_elements() - m
        M = np.zeros((2*(m+n), 2*(m+n)), dtype=np.complex64)

        # Compute the top half of the "big matrix".
        for i in range(m+n):
            p = self.centers[i]
            for j in range(m+n):
                qa, qb = self.geometry.edge_vertices(j)

                element_m = m_rad(k, p, qa, qb, i == j)
                element_l = l_rad(k, p, qa, qb, i == j)

                M[i,         j] = -element_m
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
                qa, qb = self.geometry.edge_vertices(j)
                element_l = l_rad(solution.k, p, qa, qb, False)
                element_m = m_rad(solution.k, p, qa, qb, False)
                sum += element_l * solution.velocities[j] - element_m * solution.phis[j]
            phis[i] = sum

        return SampleSolution(solution, phis)

    def solve_exterior(self, solution, samples):
        phis = np.empty(samples.shape[0], dtype=complex)

        for i in range(samples.shape[0]):
            p = samples[i, :]
            sum = 0.0
            for j in range(self.open_elements):
                qa, qb = self.geometry.edge_vertices(j)
                element_l = l_rad(solution.k, p, qa, qb, False)
                sum += -2.0 * element_l * solution.velocities[j]
            phis[i] = sum

        return SampleSolution(solution, phis)
