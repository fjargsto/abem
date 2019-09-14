import numpy as np
from .helmholtz_solver import HelmholtzSolver

bOptimized = True
if bOptimized:
    from .helmholtz_integrals_rad_c import compute_l, compute_m, compute_mt, compute_n
else:
    from .helmholtz_integrals_rad import compute_l, compute_m, compute_mt, compute_n


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

                element_l = compute_l(k, center, qa, qb, i == j)
                element_m = compute_m(k, center, qa, qb, i == j)
                element_mt = compute_mt(k, center, normal, qa, qb, i == j)
                element_n = compute_n(k, center, normal, qa, qb, i == j)
                
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

                element_l = compute_l(solution.k, p, qa, qb, False)
                element_m = compute_m(solution.k, p, qa, qb, False)

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

