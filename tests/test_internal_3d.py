import abem as ab
import numpy as np
from numpy.linalg import norm
from vector_compare import assert_almost_equal


def initialize(frequency):
    k = ab.frequency_to_wavenumber(frequency)
    interior_points = np.array(
        [
            [0.500, 0.000, 0.000],
            [0.000, 0.000, 0.010],
            [0.000, 0.000, 0.250],
            [0.000, 0.000, 0.500],
            [0.000, 0.000, 0.750],
        ],
        dtype=np.float32,
    )
    solver = ab.InteriorHelmholtzSolver3D(ab.sphere())
    boundary_incidence = ab.BoundaryIncidence(solver.len())
    boundary_incidence.phi.fill(0.0)
    boundary_incidence.v.fill(0.0)
    interior_incident_phi = np.zeros(interior_points.shape[0], dtype=np.complex64)
    return k, interior_points, boundary_incidence, solver, interior_incident_phi


def test_internal_3d_1():
    # Test Problem 1
    # Dirichlet boundary condition with phi = sin(k*z)
    #
    frequency = 20.0  # frequency [Hz]
    k, interior_points, boundary_incidence, solver, interior_incident_phi = initialize(
        frequency
    )
    boundary_condition = solver.dirichlet_boundary_condition()
    boundary_condition.f[:] = np.sin(k * solver.centers[:, 2])

    boundary_solution = solver.solve_boundary(k, boundary_condition, boundary_incidence)
    sample_solution = boundary_solution.solve_samples(
        interior_incident_phi, interior_points
    )

    phi_golden = [
        0.2804e00 + 0.0000e00j,
        0.2804e00 + 0.0000e00j,
        0.2804e00 + 0.0000e00j,
        0.2804e00 + 0.0000e00j,
        0.2804e00 + 0.0000e00j,
        0.2804e00 + 0.0000e00j,
        0.1617e00 + 0.0000e00j,
        0.8113e-01 + 0.0000e00j,
        0.1617e00 + 0.0000e00j,
        0.8113e-01 + 0.0000e00j,
        0.1617e00 + 0.0000e00j,
        0.8113e-01 + 0.0000e00j,
        0.1617e00 + 0.0000e00j,
        0.8113e-01 + 0.0000e00j,
        0.1617e00 + 0.0000e00j,
        0.8113e-01 + 0.0000e00j,
        0.1617e00 + 0.0000e00j,
        0.8113e-01 + 0.0000e00j,
        -0.1617e00 + 0.0000e00j,
        -0.8113e-01 + 0.0000e00j,
        -0.1617e00 + 0.0000e00j,
        -0.8113e-01 + 0.0000e00j,
        -0.1617e00 + 0.0000e00j,
        -0.8113e-01 + 0.0000e00j,
        -0.1617e00 + 0.0000e00j,
        -0.8113e-01 + 0.0000e00j,
        -0.1617e00 + 0.0000e00j,
        -0.8113e-01 + 0.0000e00j,
        -0.1617e00 + 0.0000e00j,
        -0.8113e-01 + 0.0000e00j,
        -0.2804e00 + 0.0000e00j,
        -0.2804e00 + 0.0000e00j,
        -0.2804e00 + 0.0000e00j,
        -0.2804e00 + 0.0000e00j,
        -0.2804e00 + 0.0000e00j,
        -0.2804e00 + 0.0000e00j,
    ]
    assert_almost_equal(phi_golden, boundary_solution.phis, 0.0001)

    phi_golden = [
        -0.1032e-15 - 0.5254e-16j,
        0.3568e-02 - 0.3302e-04j,
        0.8894e-01 - 0.8015e-03j,
        0.1757e00 - 0.1477e-02j,
        0.2623e00 - 0.1966e-02j,
    ]
    assert_almost_equal(phi_golden, sample_solution.phis, 0.00001)


def test_internal_3d_2():
    # Test Problem 2
    # Neumann boundary condition with v = cos(k*z)
    #
    frequency = 20.0  # frequency [Hz]
    k, interior_points, boundary_incidence, solver, interior_incident_phi = initialize(
        frequency
    )
    boundary_condition = solver.neumann_boundary_condition()

    for i in range(solver.len()):
        a, b, c = solver.geometry.triangle_vertices(i)
        normal = ab.normal_3d(a, b, c)
        boundary_condition.f[i] = k * np.cos(k * solver.centers[i, 2]) * normal[2]

    boundary_solution = solver.solve_boundary(k, boundary_condition, boundary_incidence)
    sample_solution = boundary_solution.solve_samples(
        interior_incident_phi, interior_points
    )

    phi_golden = [
        0.2543e00 + 0.5479e-02j,
        0.2543e00 + 0.5477e-02j,
        0.2543e00 + 0.5479e-02j,
        0.2543e00 + 0.5479e-02j,
        0.2543e00 + 0.5477e-02j,
        0.2543e00 + 0.5479e-02j,
        0.1439e00 + 0.4738e-02j,
        0.6877e-01 + 0.3551e-02j,
        0.1439e00 + 0.4748e-02j,
        0.6877e-01 + 0.3551e-02j,
        0.1439e00 + 0.4738e-02j,
        0.6877e-01 + 0.3554e-02j,
        0.1439e00 + 0.4738e-02j,
        0.6877e-01 + 0.3551e-02j,
        0.1439e00 + 0.4748e-02j,
        0.6877e-01 + 0.3551e-02j,
        0.1439e00 + 0.4738e-02j,
        0.6877e-01 + 0.3554e-02j,
        -0.1439e00 - 0.4738e-02j,
        -0.6877e-01 - 0.3551e-02j,
        -0.1439e00 - 0.4748e-02j,
        -0.6877e-01 - 0.3551e-02j,
        -0.1439e00 - 0.4738e-02j,
        -0.6877e-01 - 0.3554e-02j,
        -0.1439e00 - 0.4738e-02j,
        -0.6877e-01 - 0.3551e-02j,
        -0.1439e00 - 0.4748e-02j,
        -0.6877e-01 - 0.3551e-02j,
        -0.1439e00 - 0.4738e-02j,
        -0.6877e-01 - 0.3554e-02j,
        -0.2543e00 - 0.5479e-02j,
        -0.2543e00 - 0.5477e-02j,
        -0.2543e00 - 0.5479e-02j,
        -0.2543e00 - 0.5479e-02j,
        -0.2543e00 - 0.5477e-02j,
        -0.2543e00 - 0.5479e-02j,
    ]
    assert_almost_equal(phi_golden, boundary_solution.phis, 0.0001)

    phi_golden = [
        -0.7286e-16 - 0.6075e-15j,
        0.3205e-02 + 0.5491e-04j,
        0.8002e-01 + 0.1338e-02j,
        0.1586e00 + 0.2495e-02j,
        0.2377e00 + 0.3530e-02j,
    ]
    assert_almost_equal(phi_golden, sample_solution.phis, 0.0001)


def test_internal_3d_3():
    # Test Problem 3
    # Neumann boundary condition with v = cos(k*z)
    #
    frequency = 20.0  # frequency [Hz]
    k, interior_points, boundary_incidence, solver, interior_incident_phi = initialize(
        frequency
    )
    boundary_condition = solver.neumann_boundary_condition()

    p = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    for i in range(solver.len()):
        q = solver.centers[i, :]
        r = p - q
        R = norm(r)
        boundary_incidence.phi[i] = np.exp(1.0j * k * R) / (4.0 * np.pi * R)
        a, b, c = solver.geometry.triangle_vertices(i)
        normal = ab.normal_3d(a, b, c)
        drbdn = -np.dot(r, normal) / R
        boundary_incidence.v[i] = (
            drbdn * np.exp(1.0j * k * R) * (1.0j * k * R - 1.0) / (4.0 * np.pi * R * R)
        )

    for i in range(interior_points.shape[0]):
        q = interior_points[i, :]
        r = p - q
        R = norm(r)
        interior_incident_phi[i] = np.exp(1.0j * k * R) / (4.0 * np.pi * R)

    boundary_solution = solver.solve_boundary(k, boundary_condition, boundary_incidence)
    sample_solution = boundary_solution.solve_samples(
        interior_incident_phi, interior_points
    )

    phi_golden = [
        -0.1398e01 + -0.5759e00j,
        -0.1398e01 + -0.5759e00j,
        -0.1398e01 + -0.5759e00j,
        -0.1398e01 + -0.5759e00j,
        -0.1398e01 + -0.5759e00j,
        -0.1398e01 + -0.5759e00j,
        -0.1399e01 + -0.5779e00j,
        -0.1389e01 + -0.5760e00j,
        -0.1399e01 + -0.5779e00j,
        -0.1389e01 + -0.5760e00j,
        -0.1399e01 + -0.5779e00j,
        -0.1389e01 + -0.5760e00j,
        -0.1399e01 + -0.5779e00j,
        -0.1389e01 + -0.5760e00j,
        -0.1399e01 + -0.5779e00j,
        -0.1389e01 + -0.5760e00j,
        -0.1399e01 + -0.5779e00j,
        -0.1389e01 + -0.5760e00j,
        -0.1399e01 + -0.5779e00j,
        -0.1389e01 + -0.5760e00j,
        -0.1399e01 + -0.5779e00j,
        -0.1389e01 + -0.5760e00j,
        -0.1399e01 + -0.5779e00j,
        -0.1389e01 + -0.5760e00j,
        -0.1399e01 + -0.5779e00j,
        -0.1389e01 + -0.5760e00j,
        -0.1399e01 + -0.5779e00j,
        -0.1389e01 + -0.5760e00j,
        -0.1399e01 + -0.5779e00j,
        -0.1389e01 + -0.5760e00j,
        -0.1398e01 + -0.5759e00j,
        -0.1398e01 + -0.5759e00j,
        -0.1398e01 + -0.5759e00j,
        -0.1398e01 + -0.5759e00j,
        -0.1398e01 + -0.5759e00j,
        -0.1398e01 + -0.5759e00j,
    ]
    assert_almost_equal(phi_golden, boundary_solution.phis, 0.001)

    phi_golden = [
        -0.1291e01 - 0.5891e00j,
        0.6496e01 - 0.5949e00j,
        -0.1143e01 - 0.5941e00j,
        -0.1296e01 - 0.5904e00j,
        -0.1372e01 - 0.5989e00j,
    ]

    assert_almost_equal(phi_golden, sample_solution.phis, 0.001)
