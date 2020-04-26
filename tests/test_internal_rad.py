import abem as ab
import numpy as np
from vector_compare import assert_almost_equal


def initialize(frequency):
    k = ab.frequency_to_wavenumber(frequency)
    interior_points = np.array(
        [[0.0000, 0.0000], [0.0000, 0.5000], [0.0000, -0.5000], [0.5000, 0.0000]],
        dtype=np.float32,
    )
    interior_incident_phi = np.zeros(interior_points.shape[0], dtype=np.complex64)
    solver = ab.InteriorHelmholtzSolverRAD(ab.sphere_rad())
    boundary_incidence = ab.BoundaryIncidence(solver.len())
    boundary_incidence.phi.fill(0.0)
    boundary_incidence.v.fill(0.0)
    return k, interior_points, interior_incident_phi, solver, boundary_incidence


def test_internal_rad_1():
    frequency = 40.0  # frequency [Hz]
    k, interior_points, interior_incident_phi, solver, boundary_incidence = initialize(
        frequency
    )

    # Test Problem 1
    # Dirichlet boundary condition with phi = sin(k*z)
    #
    boundary_condition = solver.dirichlet_boundary_condition()
    boundary_condition.f[:] = np.sin(k * solver.centers[:, 1])

    boundary_solution = solver.solve_boundary(k, boundary_condition, boundary_incidence)
    sample_solution = boundary_solution.solve_samples(
        interior_incident_phi, interior_points
    )

    phi_golden = [
        0.6632e00 + 0.0000e0j,
        0.6467e00 + 0.0000e0j,
        0.6129e00 + 0.0000e0j,
        0.5615e00 + 0.0000e0j,
        0.4923e00 + 0.0000e0j,
        0.4055e00 + 0.0000e0j,
        0.3028e00 + 0.0000e0j,
        0.1874e00 + 0.0000e0j,
        0.6352e-01 + 0.0000e0j,
        -0.6352e-01 + 0.0000e0j,
        -0.1874e00 + 0.0000e0j,
        -0.3028e00 + 0.0000e0j,
        -0.4055e00 + 0.0000e0j,
        -0.4923e00 + 0.0000e0j,
        -0.5615e00 + 0.0000e0j,
        -0.6129e00 + 0.0000e0j,
        -0.6467e00 + 0.0000e0j,
        -0.6632e00 + 0.0000e0j,
    ]

    assert_almost_equal(phi_golden, boundary_solution.phis, 0.0001)

    phi_golden = [
        0.2826e-08 + -0.5606e-09j,
        0.3574e00 + -0.3513e-03j,
        -0.3574e00 + 0.3513e-03j,
        0.2897e-08 + -0.7412e-09j,
    ]

    assert_almost_equal(phi_golden, sample_solution.phis, 0.0001)


def test_internal_rad_2():
    frequency = 40.0  # frequency [Hz]
    k, interior_points, interior_incident_phi, solver, boundary_incidence = initialize(
        frequency
    )

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
    sample_solution = boundary_solution.solve_samples(
        interior_incident_phi, interior_points
    )

    phi_golden = [
        0.6564e00 + 0.2434e-02j,
        0.6397e00 + 0.2398e-02j,
        0.6063e00 + 0.2226e-02j,
        0.5555e00 + 0.1963e-02j,
        0.4870e00 + 0.1653e-02j,
        0.4012e00 + 0.1298e-02j,
        0.2996e00 + 0.9183e-03j,
        0.1854e00 + 0.5598e-03j,
        0.6281e-01 + 0.1946e-03j,
        -0.6281e-01 - 0.1946e-03j,
        -0.1854e00 - 0.5598e-03j,
        -0.2996e00 - 0.9183e-03j,
        -0.4012e00 - 0.1298e-02j,
        -0.4870e00 - 0.1653e-02j,
        -0.5555e00 - 0.1963e-02j,
        -0.6063e00 - 0.2226e-02j,
        -0.6397e00 - 0.2398e-02j,
        -0.6564e00 - 0.2434e-02j,
    ]

    assert_almost_equal(phi_golden, boundary_solution.phis, 0.001)

    phi_golden = [
        0.2234e-07 - 0.1671e-07j,
        0.3536e00 + 0.9056e-03j,
        -0.3536e00 - 0.9056e-03j,
        0.2128e-07 - 0.1615e-07j,
    ]

    assert_almost_equal(phi_golden, sample_solution.phis, 0.0001)


def test_internal_rad_3():
    frequency = 40.0  # frequency [Hz]
    k, interior_points, interior_incident_phi, solver, boundary_incidence = initialize(
        frequency
    )

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
        boundary_incidence.v[i] = (
            drbdn
            * np.exp(1j * k * rpq)
            * (1j * k * rpq - 1.0)
            / (4.0 * np.pi * rpq * rpq)
        )

    for i in range(interior_points.shape[0]):
        r = interior_points[i, 0]
        z = interior_points[i, 1]
        # make input complex so proper sqrt is called
        rpq = np.sqrt(0.0j + r ** 2 + (zp - z) ** 2)
        interior_incident_phi[i] = np.exp(1j * k * rpq) / (4.0 * np.pi * rpq)

    boundary_solution = solver.solve_boundary(k, boundary_condition, boundary_incidence)
    sample_solution = boundary_solution.solve_samples(
        interior_incident_phi, interior_points
    )

    phi_golden = [
        0.9096e-01 + 0.5529e-01j,
        0.8933e-01 + 0.5521e-01j,
        0.8636e-01 + 0.5507e-01j,
        0.8234e-01 + 0.5485e-01j,
        0.7766e-01 + 0.5457e-01j,
        0.7273e-01 + 0.5424e-01j,
        0.6779e-01 + 0.5387e-01j,
        0.6306e-01 + 0.5346e-01j,
        0.5870e-01 + 0.5305e-01j,
        0.5473e-01 + 0.5262e-01j,
        0.5120e-01 + 0.5220e-01j,
        0.4816e-01 + 0.5181e-01j,
        0.4557e-01 + 0.5145e-01j,
        0.4342e-01 + 0.5113e-01j,
        0.4173e-01 + 0.5086e-01j,
        0.4046e-01 + 0.5065e-01j,
        0.3962e-01 + 0.5051e-01j,
        0.3921e-01 + 0.5044e-01j,
    ]

    assert_almost_equal(phi_golden, boundary_solution.phis, 0.00001)

    phi_golden = [
        0.3118e00 + 0.5882e-01j,
        0.3116e00 + 0.5901e-01j,
        0.8958e-01 + 0.5608e-01j,
        0.1295e00 + 0.5751e-01j,
    ]

    assert_almost_equal(phi_golden, sample_solution.phis, 0.0001)
