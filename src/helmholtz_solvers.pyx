import numpy as np
cimport numpy as np
from solver import Solver
from acoustic_boundaries import ExteriorBoundarySolution, InteriorBoundarySolution
#from iops_pyx import * # Cython integral operators (slow!)
#from iops_sci import * # Scikit based integral operators (slow!)
from iops_cpp import * # C++ integral operators


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


# -----------------------------------------------------------------------------
# 2D Solvers
# -----------------------------------------------------------------------------
class HelmholtzSolver2D(HelmholtzSolver):
    def __init__(self, chain, c=344.0, density=1.205):
        super(HelmholtzSolver2D, self).__init__(chain, c, density)
        self.centers = self.geometry.centers()
        # length of the boundary elements (for the 3d shapes this is replaced by aArea)
        self.lengths = self.geometry.lengths()

    def compute_boundary_matrices(self, k, mu, orientation):
        A = np.empty((self.len(), self.len()), dtype=complex)
        B = np.empty(A.shape, dtype=complex)

        centers = self.geometry.centers()
        normals = -self.geometry.normals()

        for i in range(self.len()):
            center = centers[i]
            normal = normals[i]
            for j in range(self.len()):
                qa, qb = self.geometry.edge_vertices(j)

                element_l = l_2d(k, center, qa, qb, i == j)
                element_m = m_2d(k, center, qa, qb, i == j)
                element_mt = mt_2d(k, center, normal, qa, qb, i == j)
                element_n = n_2d(k, center, normal, qa, qb, i == j)

                A[i, j] = element_l + mu * element_mt
                B[i, j] = element_m + mu * element_n

            if orientation == 'interior':
                # interior variant, signs are reversed for exterior
                A[i, i] -= 0.5 * mu
                B[i, i] += 0.5
            elif orientation == 'exterior':
                A[i, i] += 0.5 * mu
                B[i, i] -= 0.5
            else:
                assert False, 'Invalid orientation: {}'.format(orientation)

        return A, B

    def compute_boundary_matrices_interior(self, k, mu):
        return self.compute_boundary_matrices(k, mu, 'interior')

    def compute_boundary_matrices_exterior(self, k, mu):
        return self.compute_boundary_matrices(k, mu, 'exterior')

    def solve_samples(self, solution, incident_phis, samples, orientation):
        assert incident_phis.shape == samples.shape[:-1], \
            "Incident phi vector and sample points vector must match"

        results = np.empty(samples.shape[0], dtype=complex)

        for i in range(incident_phis.size):
            p = samples[i]
            sum = incident_phis[i]
            for j in range(solution.phis.size):
                qa, qb = self.geometry.edge_vertices(j)

                element_l = l_2d(solution.k, p, qa, qb, False)
                element_m = m_2d(solution.k, p, qa, qb, False)
                if orientation == 'interior':
                    sum += element_l * solution.velocities[j] - element_m * solution.phis[j]
                elif orientation == 'exterior':
                    sum -= element_l * solution.velocities[j] - element_m * solution.phis[j]
                else:
                    assert False, 'Invalid orientation: {}'.format(orientation)
            results[i] = sum
        return results


class InteriorHelmholtzSolver2D(HelmholtzSolver2D):
    def solve_boundary(self, k, boundary_condition, boundary_incidence, mu=None):
        return super(InteriorHelmholtzSolver2D, self).solve_boundary('interior', k,
                                                                     boundary_condition,
                                                                     boundary_incidence, mu)


class ExteriorHelmholtzSolver2D(HelmholtzSolver2D):
    def solve_boundary(self, k, boundary_condition, boundary_incidence, mu=None):
        return super(ExteriorHelmholtzSolver2D, self).solve_boundary('exterior', k,
                                                                     boundary_condition,
                                                                     boundary_incidence, mu)


# -----------------------------------------------------------------------------
# 3D Solver
# -----------------------------------------------------------------------------
class HelmholtzSolver3D(HelmholtzSolver):
    def __init__(self, geometry, c=344.0, density=1.205):
        super(HelmholtzSolver3D, self).__init__(geometry, c, density)
        self.centers = self.geometry.centers()
        self.areas = self.geometry.areas()
        self.normals = self.geometry.normals()

    # noinspection PyPep8Naming
    def compute_boundary_matrices(self, k, mu, orientation):
        A = np.empty((self.len(), self.len()), dtype=complex)
        B = np.empty(A.shape, dtype=complex)

        for i in range(self.len()):
            p = self.centers[i]
            normal = self.normals[i]
            for j in range(self.len()):
                qa, qb, qc = self.geometry.triangle_vertices(j)

                element_l = l_3d(k, p, qa, qb, qc, i == j)
                element_m = m_3d(k, p, qa, qb, qc, i == j)
                element_mt = mt_3d(k, p, normal, qa, qb, qc, i == j)
                element_n = n_3d(k, p, normal, qa, qb, qc, i == j)

                A[i, j] = element_l + mu * element_mt
                B[i, j] = element_m + mu * element_n

            if orientation == 'interior':
                # interior variant, signs are reversed for exterior
                A[i, i] -= 0.5 * mu
                B[i, i] += 0.5
            elif orientation == 'exterior':
                A[i, i] += 0.5 * mu
                B[i, i] -= 0.5
            else:
                assert False, 'Invalid orientation: {}'.format(orientation)

        return A, B

    def compute_boundary_matrices_interior(self, k, mu):
        return self.compute_boundary_matrices(k, mu, 'interior')

    def compute_boundary_matrices_exterior(self, k, mu):
        return self.compute_boundary_matrices(k, mu, 'exterior')

    def solve_samples(self, solution, incident_phis, samples, orientation):
        assert incident_phis.shape == samples.shape[:-1], \
            "Incident phi vector and sample points vector must match"

        results = np.empty(samples.shape[0], dtype=complex)

        for i in range(incident_phis.size):
            p = samples[i]
            sum = incident_phis[i]
            for j in range(solution.phis.size):
                qa, qb, qc = self.geometry.triangle_vertices(j)
                element_l = l_3d(solution.k, p, qa, qb, qc, False)
                element_m = m_3d(solution.k, p, qa, qb, qc, False)
                if orientation == 'interior':
                    sum += element_l * solution.velocities[j] - element_m * solution.phis[j]
                elif orientation == 'exterior':
                    sum -= element_l * solution.velocities[j] - element_m * solution.phis[j]
                else:
                    assert False, 'Invalid orientation: {}'.format(orientation)
            results[i] = sum
        return results


class InteriorHelmholtzSolver3D(HelmholtzSolver3D):
    def solve_boundary(self, k, boundary_condition, boundary_incidence, mu = None):
        return super(InteriorHelmholtzSolver3D,
                     self).solve_boundary('interior', k,
                                          boundary_condition, boundary_incidence,
                                          mu)


class ExteriorHelmholtzSolver3D(HelmholtzSolver3D):
    def solve_boundary(self, k, boundary_condition, boundary_incidence, mu = None):
        return super(ExteriorHelmholtzSolver3D,
                     self).solve_boundary('exterior', k,
                                          boundary_condition, boundary_incidence,
                                          mu)


# -----------------------------------------------------------------------------
# RAD Solvers
# -----------------------------------------------------------------------------
class HelmholtzSolverRAD(HelmholtzSolver):
    def __init__(self, chain, c=344.0, density=1.205):
        super(HelmholtzSolverRAD, self).__init__(chain, c, density)
        self.centers = self.geometry.centers()
        # area of the boundary elements
        self.area = self.geometry.areas()

    def compute_boundary_matrices(self, k, mu, orientation):
        A = np.empty((self.len(), self.len()), dtype=complex)
        B = np.empty(A.shape, dtype=complex)

        centers = self.geometry.centers()
        normals = self.geometry.normals()

        for i in range(self.len()):
            center = centers[i]
            normal = -normals[i]
            for j in range(self.len()):
                qa, qb = self.geometry.edge_vertices(j)

                element_l = l_rad(k, center, qa, qb, i == j)
                element_m = m_rad(k, center, qa, qb, i == j)
                element_mt = mt_rad(k, center, normal, qa, qb, i == j)
                element_n = n_rad(k, center, normal, qa, qb, i == j)

                A[i, j] = element_l + mu * element_mt
                B[i, j] = element_m + mu * element_n

            if orientation == 'interior':
                A[i, i] -= 0.5 * mu
                B[i, i] += 0.5
            elif orientation == 'exterior':
                A[i, i] += 0.5 * mu
                B[i, i] -= 0.5
            else:
                assert False, 'Invalid orientation: {}'.format(orientation)

        return A, B

    def compute_boundary_matrices_interior(self, k, mu):
        return self.compute_boundary_matrices(k, mu, 'interior')

    def compute_boundary_matrices_exterior(self, k, mu):
        return self.compute_boundary_matrices(k, mu, 'exterior')

    def solve_samples(self, solution, incident_phis, samples, orientation):
        assert incident_phis.shape == samples.shape[:-1], \
            "Incident phi vector and samples vector must match"

        results = np.empty(samples.shape[0], dtype=complex)
        for i in range(incident_phis.size):
            p = samples[i]
            sum = incident_phis[i]
            for j in range(solution.phis.size):
                qa, qb = self.geometry.edge_vertices(j)

                element_l = l_rad(solution.k, p, qa, qb, False)
                element_m = m_rad(solution.k, p, qa, qb, False)

                if orientation == 'interior':
                    sum += element_l * solution.velocities[j] - element_m * solution.phis[j]
                elif orientation == 'exterior':
                    sum -= element_l * solution.velocities[j] - element_m * solution.phis[j]
                else:
                    assert False, 'Invalid orientation: {}'.format(orientation)
            results[i] = sum
        return results


class InteriorHelmholtzSolverRAD(HelmholtzSolverRAD):
    def solve_boundary(self, k, boundary_condition, boundary_incidence, mu = None):
        return super(InteriorHelmholtzSolverRAD,
                     self).solve_boundary('interior', k,
                                          boundary_condition, boundary_incidence,
                                          mu)


class ExteriorHelmholtzSolverRAD(HelmholtzSolverRAD):
    def solve_boundary(self, k, boundary_condition, boundary_incidence, mu = None):
        return super(ExteriorHelmholtzSolverRAD,
                     self).solve_boundary('exterior', k,
                                          boundary_condition, boundary_incidence,
                                          mu)

