import numpy as np
from .helmholtz_solver import HelmholtzSolver

from .helmholtz_integrals_2d import l_2d, m_2d, mt_2d, n_2d


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
