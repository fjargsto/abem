import abem as ab
import numpy as np
from .vector_compare import assert_almost_equal


def initialize(frequency):
    k = ab.frequency_to_wavenumber(frequency)
    interior_points = np.array([[0.0000, 0.0000],
                                [0.0000, 0.5000],
                                [0.0000, -0.5000],
                                [0.5000, 0.0000]], dtype=np.float32)
    interior_incident_phi = np.zeros(interior_points.shape[0], dtype=np.complex64)
    solver = ab.InteriorHelmholtzSolverRAD(ab.sphere_rad())
    boundary_incidence = ab.BoundaryIncidence(solver.len())
    boundary_incidence.phi.fill(0.0)
    boundary_incidence.v.fill(0.0)
    return k, interior_points, interior_incident_phi, solver, boundary_incidence


def test_internal_rad_1():
    frequency = 40.0  # frequency [Hz]
    k, interior_points, interior_incident_phi, solver, boundary_incidence = initialize(frequency)

    # Test Problem 1
    # Dirichlet boundary condition with phi = sin(k*z)
    #
    boundary_condition = solver.dirichlet_boundary_condition()
    boundary_condition.f[:] = np.sin(k * solver.centers[:, 1])

    boundary_solution = solver.solve_boundary(k, boundary_condition, boundary_incidence)
    sample_solution = boundary_solution.solve_samples(interior_incident_phi, interior_points)

    phi_golden = [0.6632E+00 + 0.0000E+0j,
                  0.6467E+00 + 0.0000E+0j,
                  0.6129E+00 + 0.0000E+0j,
                  0.5615E+00 + 0.0000E+0j,
                  0.4923E+00 + 0.0000E+0j,
                  0.4055E+00 + 0.0000E+0j,
                  0.3028E+00 + 0.0000E+0j,
                  0.1874E+00 + 0.0000E+0j,
                  0.6352E-01 + 0.0000E+0j,
                  -.6352E-01 + 0.0000E+0j,
                  -.1874E+00 + 0.0000E+0j,
                  -.3028E+00 + 0.0000E+0j,
                  -.4055E+00 + 0.0000E+0j,
                  -.4923E+00 + 0.0000E+0j,
                  -.5615E+00 + 0.0000E+0j,
                  -.6129E+00 + 0.0000E+0j,
                  -.6467E+00 + 0.0000E+0j,
                  -.6632E+00 + 0.0000E+0j, ]

    assert_almost_equal(phi_golden, boundary_solution.phis, 0.0001)

    phi_golden = [0.2826E-08+ -.5606E-09j,
                  0.3574E+00+ -.3513E-03j,
                  -.3574E+00+ 0.3513E-03j,
                  0.2897E-08+ -.7412E-09j,]

    assert_almost_equal(phi_golden, sample_solution.phis, 0.0001)


def test_internal_rad_2():
    frequency = 40.0  # frequency [Hz]
    k, interior_points, interior_incident_phi, solver, boundary_incidence = initialize(frequency)

    # Test Problem 2
    # von Neumann boundary condition such that phi = sin(k/sqrt(2) * x) * sin(k/sqrt(2) * y)
    # Differentiate with respect to x and y to obtain outward normal
    boundary_condition = solver.neumann_boundary_condition()
    centers = solver.geometry.centers()
    normals = solver.geometry.normals()
    for i in range(centers.shape[0]):
        z = centers[i, 1]
        n = -normals[i]
        boundary_condition.f[i] = k * np.cos(k * z) * n[1]

    boundary_solution = solver.solve_boundary(k, boundary_condition, boundary_incidence)
    sample_solution = boundary_solution.solve_samples(interior_incident_phi, interior_points)

    phi_golden = [ 0.6564E+00 + 0.2434E-02j,
                   0.6397E+00 + 0.2398E-02j,
                   0.6063E+00 + 0.2226E-02j,
                   0.5555E+00 + 0.1963E-02j,
                   0.4870E+00 + 0.1653E-02j,
                   0.4012E+00 + 0.1298E-02j,
                   0.2996E+00 + 0.9183E-03j,
                   0.1854E+00 + 0.5598E-03j,
                   0.6281E-01 + 0.1946E-03j,
                  -0.6281E-01 - 0.1946E-03j,
                  -0.1854E+00 - 0.5598E-03j,
                  -0.2996E+00 - 0.9183E-03j,
                  -0.4012E+00 - 0.1298E-02j,
                  -0.4870E+00 - 0.1653E-02j,
                  -0.5555E+00 - 0.1963E-02j,
                  -0.6063E+00 - 0.2226E-02j,
                  -0.6397E+00 - 0.2398E-02j,
                  -0.6564E+00 - 0.2434E-02j,]

    assert_almost_equal(phi_golden, boundary_solution.phis, 0.001)

    phi_golden = [ 0.2234E-07 - 0.1671E-07j,
                   0.3536E+00 + 0.9056E-03j,
                  -0.3536E+00 - 0.9056E-03j,
                   0.2128E-07 - 0.1615E-07j,]

    assert_almost_equal(phi_golden, sample_solution.phis, 0.0001)

def test_internal_rad_3():
    frequency = 40.0  # frequency [Hz]
    k, interior_points, interior_incident_phi, solver, boundary_incidence = initialize(frequency)

    # Test Problem 3
    # Dirichlet boundary condition, such that phi = sin(k/ sqrt(2) * x) * sin(k/sqrt(2) * y)
    # Differentiate with respect to x and y to obtain outward normal
    boundary_condition = solver.neumann_boundary_condition()
    boundary_condition.alpha.fill(1.0)
    boundary_condition.beta.fill(0.0)

    centers = solver.geometry.centers()
    normals = solver.geometry.normals()

    zp = 0.25
    for i in range(centers.shape[0]):
        r = centers[i, 0]
        z = centers[i, 1]
        # make input complex so proper sqrt is called
        rpq = np.sqrt(0.0j + r ** 2 + (z - zp) ** 2)
        boundary_condition.f[i] = np.exp(1j * k * rpq) / (4.0 * np.pi * rpq)
        boundary_incidence.phi[i] = np.exp(1j * k * rpq) / (4.0 * np.pi * rpq)
        n = -normals[i]
        drbdn = (r * n[0] + (z - zp) * n[1]) / rpq
        boundary_incidence.v[i] = drbdn * np.exp(1j * k * rpq) * (1j * k * rpq - 1.0) \
                                  / (4.0 * np.pi * rpq * rpq)

    for i in range(interior_points.shape[0]):
        r = interior_points[i, 0]
        z = interior_points[i, 1]
        # make input complex so proper sqrt is called
        rpq = np.sqrt(0.0j + r ** 2 + (zp - z) ** 2)
        interior_incident_phi[i] = np.exp(1j * k * rpq) / (4.0 * np.pi * rpq)

    boundary_solution = solver.solve_boundary(k, boundary_condition, boundary_incidence)
    sample_solution = boundary_solution.solve_samples(interior_incident_phi, interior_points)

    phi_golden = [0.9096E-01 + 0.5529E-01j,
                  0.8933E-01 + 0.5521E-01j,
                  0.8636E-01 + 0.5507E-01j,
                  0.8234E-01 + 0.5485E-01j,
                  0.7766E-01 + 0.5457E-01j,
                  0.7273E-01 + 0.5424E-01j,
                  0.6779E-01 + 0.5387E-01j,
                  0.6306E-01 + 0.5346E-01j,
                  0.5870E-01 + 0.5305E-01j,
                  0.5473E-01 + 0.5262E-01j,
                  0.5120E-01 + 0.5220E-01j,
                  0.4816E-01 + 0.5181E-01j,
                  0.4557E-01 + 0.5145E-01j,
                  0.4342E-01 + 0.5113E-01j,
                  0.4173E-01 + 0.5086E-01j,
                  0.4046E-01 + 0.5065E-01j,
                  0.3962E-01 + 0.5051E-01j,
                  0.3921E-01 + 0.5044E-01j,]

    assert_almost_equal(phi_golden, boundary_solution.phis, 0.00001)

    phi_golden = [0.3118E+00 + 0.5882E-01j,
                  0.3116E+00 + 0.5901E-01j,
                  0.8958E-01 + 0.5608E-01j,
                  0.1295E+00 + 0.5751E-01j,]

    assert_almost_equal(phi_golden, sample_solution.phis, 0.0001)

