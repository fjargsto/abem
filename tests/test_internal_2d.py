import abem as ab
import numpy as np
from scipy.special import hankel1
from numpy.linalg import norm
from vector_compare import assert_almost_equal


def initialize(frequency):
    k = ab.frequency_to_wavenumber(frequency)
    solver = ab.InteriorHelmholtzSolver2D(ab.square())
    interior_points = np.array(
        [
            [0.0250, 0.0250],
            [0.0750, 0.0250],
            [0.0250, 0.0750],
            [0.0750, 0.0750],
            [0.0500, 0.0500],
        ],
        dtype=np.float32,
    )

    interior_incident_phi = np.zeros(interior_points.shape[0], dtype=np.complex64)

    boundary_incidence = ab.BoundaryIncidence(solver.len())
    boundary_incidence.phi.fill(0.0)
    boundary_incidence.v.fill(0.0)

    return (k, interior_points, interior_incident_phi, solver, boundary_incidence)


def test_interal_2d_1():
    frequency = 400.0  # frequency [Hz]
    (
        k,
        interior_points,
        interior_incident_phi,
        solver,
        boundary_incidence,
    ) = initialize(frequency)

    # Test Problem 1
    # Dirichlet boundary condition with phi = sin(k/sqrt(2)*x) * sin(k/sqrt(2)*y)
    #
    boundary_condition = solver.dirichlet_boundary_condition()
    boundary_condition.f[:] = np.sin(k / np.sqrt(2.0) * solver.centers[:, 0]) * np.sin(
        k / np.sqrt(2.0) * solver.centers[:, 1]
    )

    boundary_solution = solver.solve_boundary(k, boundary_condition, boundary_incidence)
    sample_solution = boundary_solution.solve_samples(
        interior_incident_phi, interior_points
    )

    phi_golden = [
        0.0000e00 + 0.0000e00j,
        0.0000e00 + 0.0000e00j,
        0.0000e00 + 0.0000e00j,
        0.0000e00 + 0.0000e00j,
        0.0000e00 + 0.0000e00j,
        0.0000e00 + 0.0000e00j,
        0.0000e00 + 0.0000e00j,
        0.0000e00 + 0.0000e00j,
        0.1595e-01 + 0.0000e00j,
        0.4777e-01 + 0.0000e00j,
        0.7940e-01 + 0.0000e00j,
        0.1107e00 + 0.0000e00j,
        0.1415e00 + 0.0000e00j,
        0.1718e00 + 0.0000e00j,
        0.2013e00 + 0.0000e00j,
        0.2300e00 + 0.0000e00j,
        0.2300e00 + 0.0000e00j,
        0.2013e00 + 0.0000e00j,
        0.1718e00 + 0.0000e00j,
        0.1415e00 + 0.0000e00j,
        0.1107e00 + 0.0000e00j,
        0.7940e-01 + 0.0000e00j,
        0.4777e-01 + 0.0000e00j,
        0.1595e-01 + 0.0000e00j,
        0.0000e00 + 0.0000e00j,
        0.0000e00 + 0.0000e00j,
        0.0000e00 + 0.0000e00j,
        0.0000e00 + 0.0000e00j,
        0.0000e00 + 0.0000e00j,
        0.0000e00 + 0.0000e00j,
        0.0000e00 + 0.0000e00j,
        0.0000e00 + 0.0000e00j,
    ]
    assert_almost_equal(phi_golden, boundary_solution.phis, 0.0001)
"""
    phi_golden = [
        0.1589e-01 + 0.1183e-03j,
        0.4818e-01 + 0.4001e-04j,
        0.4818e-01 + 0.4001e-04j,
        0.1434e00 - 0.2577e-03j,
        0.6499e-01 - 0.1422e-04j,
    ]
    assert_almost_equal(phi_golden, sample_solution.phis, 0.0001)
"""

def test_internal_2d_2():
    # Test Problem 2
    # von Neumann boundary condition such that phi = sin(k/sqrt(2) * x) * sin(k/sqrt(2) * y)
    # Differentiate with respect to x and y to obtain outward normal:
    # dPhi/dX = k/sqrt(2) * cos(k/sqrt(2) * x) * sin(k/sqrt(2) * y)
    # dPhi/dY = k/sqrt(2) * sin(k/sqrt(2) * x) * cos(k/sqrt(2) * y)
    frequency = 400.0  # frequency [Hz]
    (
        k,
        interior_points,
        interior_incident_phi,
        solver,
        boundary_incidence,
    ) = initialize(frequency)

    boundary_condition = solver.neumann_boundary_condition()
    w = k / np.sqrt(2.0)
    for i in range(solver.centers.shape[0]):
        x = solver.centers[i, 0]
        y = solver.centers[i, 1]
        if x < 1e-7:
            boundary_condition.f[i] = -w * np.cos(w * x) * np.sin(w * y)
        elif x > 0.1 - 1e-7:
            boundary_condition.f[i] = w * np.cos(w * x) * np.sin(w * y)
        elif y < 1e-7:
            boundary_condition.f[i] = -w * np.sin(w * x) * np.cos(w * y)
        else:
            boundary_condition.f[i] = w * np.sin(w * x) * np.cos(w * y)

    boundary_solution = solver.solve_boundary(k, boundary_condition, boundary_incidence)
    sample_solution = boundary_solution.solve_samples(
        interior_incident_phi, interior_points
    )

    phi_golden = [
        -0.3647e-03 + 0.8013e-03j,
        -0.3205e-03 + 0.8641e-03j,
        -0.2681e-03 + 0.9276e-03j,
        -0.1668e-03 + 0.9850e-03j,
        0.3219e-04 + 0.1031e-02j,
        0.4063e-03 + 0.1060e-02j,
        0.1118e-02 + 0.1066e-02j,
        0.2859e-02 + 0.1050e-02j,
        0.1937e-01 + 0.9291e-03j,
        0.4881e-01 + 0.1150e-02j,
        0.7902e-01 + 0.1426e-02j,
        0.1092e00 + 0.1723e-02j,
        0.1388e00 + 0.2028e-02j,
        0.1677e00 + 0.2333e-02j,
        0.1954e00 + 0.2627e-02j,
        0.2205e00 + 0.2877e-02j,
        0.2205e00 + 0.2877e-02j,
        0.1954e00 + 0.2627e-02j,
        0.1677e00 + 0.2333e-02j,
        0.1388e00 + 0.2028e-02j,
        0.1092e00 + 0.1723e-02j,
        0.7902e-01 + 0.1426e-02j,
        0.4881e-01 + 0.1150e-02j,
        0.1937e-01 + 0.9291e-03j,
        0.2859e-02 + 0.1050e-02j,
        0.1118e-02 + 0.1066e-02j,
        0.4063e-03 + 0.1060e-02j,
        0.3219e-04 + 0.1031e-02j,
        -0.1668e-03 + 0.9850e-03j,
        -0.2681e-03 + 0.9276e-03j,
        -0.3205e-03 + 0.8641e-03j,
        -0.3647e-03 + 0.8013e-03j,
    ]
    assert_almost_equal(phi_golden, boundary_solution.phis, 0.0001)
"""
    phi_golden = [
        0.1559e-01 + 0.1166e-02j,
        0.4805e-01 + 0.1369e-02j,
        0.4805e-01 + 0.1369e-02j,
        0.1398e00 + 0.1898e-02j,
        0.6392e-01 + 0.1475e-02j,
    ]
    assert_almost_equal(phi_golden, sample_solution.phis, 0.0001)
"""

def test_internal_2d_3():
    # Test Problem 3
    # The test problem computes the field produced by a unit source at
    # the point (0.5,0.25) within the square with a rigid boundary.
    # The rigid boundary implies the boundary condition v=0.
    # The test problem computes the field produced by a unit source at
    # the point (0.5,0.25) within the square with a rigid boundary.
    # The incident velocity potential is given by {\phi}_inc=i*h0(kr)/4
    # where r is the distance from the point (0.5,0.25)
    frequency = 400.0  # frequency [Hz]
    k, interior_points, interior_incident_phi, solver, boundary_incidence = initialize(
        frequency
    )

    boundary_condition = solver.neumann_boundary_condition()
    boundary_condition.f.fill(0.0)

    p = np.array([0.05, 0.025], dtype=np.float32)
    for i in range(solver.centers.shape[0]):
        r = solver.centers[i] - p
        R = norm(r)
        boundary_incidence.phi[i] = 0.25j * hankel1(0, k * R)
        if solver.centers[i, 0] < 1e-7:
            boundary_incidence.v[i] = -0.25j * k * hankel1(1, k * R) * (-r[0] / R)
        elif solver.centers[i, 0] > 0.1 - 1e-7:
            boundary_incidence.v[i] = -0.25j * k * hankel1(1, k * R) * (r[0] / R)
        elif solver.centers[i, 1] < 1e-7:
            boundary_incidence.v[i] = -0.25j * k * hankel1(1, k * R) * (-r[1] / R)
        elif solver.centers[i, 1] > 0.1 - 1e-7:
            boundary_incidence.v[i] = -0.25j * k * hankel1(1, k * R) * (r[1] / R)
        else:
            assert False, "All cases must be handled above."

    for i in range(interior_incident_phi.size):
        r = interior_points[i] - p
        R = norm(r)
        interior_incident_phi[i] = 0.25j * hankel1(0, k * R)

    boundary_solution = solver.solve_boundary(k, boundary_condition, boundary_incidence)
    sample_solution = boundary_solution.solve_samples(
        interior_incident_phi, interior_points
    )
    print("Test3 boundary solution")
    print(boundary_solution)
    print("Test3 sample solution")
    print(sample_solution)

    phi_golden = [
        -0.1813e01 + 0.3210e-03j,
        -0.1827e01 + 0.2720e-03j,
        -0.1855e01 + 0.2822e-03j,
        -0.1895e01 + 0.3356e-03j,
        -0.1938e01 + 0.4039e-03j,
        -0.1976e01 + 0.4646e-03j,
        -0.2004e01 + 0.5079e-03j,
        -0.2019e01 + 0.5301e-03j,
        -0.2020e01 + 0.4892e-03j,
        -0.2018e01 + 0.4470e-03j,
        -0.2016e01 + 0.4173e-03j,
        -0.2014e01 + 0.4012e-03j,
        -0.2014e01 + 0.4012e-03j,
        -0.2016e01 + 0.4173e-03j,
        -0.2018e01 + 0.4470e-03j,
        -0.2020e01 + 0.4892e-03j,
        -0.2019e01 + 0.5301e-03j,
        -0.2004e01 + 0.5079e-03j,
        -0.1976e01 + 0.4646e-03j,
        -0.1938e01 + 0.4039e-03j,
        -0.1895e01 + 0.3356e-03j,
        -0.1855e01 + 0.2822e-03j,
        -0.1827e01 + 0.2720e-03j,
        -0.1813e01 + 0.3210e-03j,
        -0.1810e01 + 0.4337e-03j,
        -0.1782e01 + 0.3972e-03j,
        -0.1731e01 + 0.2555e-03j,
        -0.1681e01 + 0.3855e-04j,
        -0.1681e01 + 0.3855e-04j,
        -0.1731e01 + 0.2555e-03j,
        -0.1782e01 + 0.3972e-03j,
        -0.1810e01 + 0.4337e-03j,
    ]
    assert_almost_equal(phi_golden, boundary_solution.phis, 0.00001)
"""
    phi_golden = [
        -0.1777e01 + 0.3587e-03j,
        -0.1777e01 + 0.3587e-03j,
        -0.1984e01 + 0.4169e-03j,
        -0.1984e01 + 0.4169e-03j,
        -0.1845e01 + 0.3876e-03j,
    ]
    assert_almost_equal(phi_golden, sample_solution.phis, 0.003)
"""