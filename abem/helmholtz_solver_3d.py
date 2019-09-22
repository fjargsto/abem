import numpy as np
from .helmholtz_solver import HelmholtzSolver
from .helmholtz_integrals_3d import l_3d, m_3d, mt_3d, n_3d

    
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
