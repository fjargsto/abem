import abem as ab
import numpy as np
from scipy.special import hankel1
from numpy.linalg import norm
from .vector_compare import assert_almost_equal


def initialize(frequency):
    k = ab.frequency_to_wavenumber(frequency)
    solver = ab.InteriorHelmholtzSolver2D(ab.square())
    interior_points = np.array([[0.0250, 0.0250],
                                [0.0750, 0.0250],
                                [0.0250, 0.0750],
                                [0.0750, 0.0750],
                                [0.0500, 0.0500]], dtype=np.float32)

    interior_incident_phi = np.zeros(interior_points.shape[0], dtype=np.complex64)

    boundary_incidence = ab.BoundaryIncidence(solver.len())
    boundary_incidence.phi.fill(0.0)
    boundary_incidence.v.fill(0.0)

    return k, interior_points, interior_incident_phi, solver, boundary_incidence

def test_interal_2d_1():
    frequency = 400.0  # frequency [Hz]
    k, interior_points, interior_incident_phi, solver, boundary_incidence = initialize(frequency)

    # Test Problem 1
    # Dirichlet boundary condition with phi = sin(k/sqrt(2)*x) * sin(k/sqrt(2)*y)
    #
    boundary_condition = solver.dirichlet_boundary_condition()
    boundary_condition.f[:] = np.sin(k / np.sqrt(2.0) * solver.centers[:, 0]) \
                              * np.sin(k / np.sqrt(2.0) * solver.centers[:, 1])

    boundary_solution = solver.solve_boundary(k, boundary_condition, boundary_incidence)
    sample_solution = boundary_solution.solve_samples(interior_incident_phi, interior_points)

    phi_golden = [0.0000E+00 + 0.0000E+00j, 0.0000E+00 + 0.0000E+00j,
                  0.0000E+00 + 0.0000E+00j, 0.0000E+00 + 0.0000E+00j,
                  0.0000E+00 + 0.0000E+00j, 0.0000E+00 + 0.0000E+00j,
                  0.0000E+00 + 0.0000E+00j, 0.0000E+00 + 0.0000E+00j,
                  0.1595E-01 + 0.0000E+00j, 0.4777E-01 + 0.0000E+00j,
                  0.7940E-01 + 0.0000E+00j, 0.1107E+00 + 0.0000E+00j,
                  0.1415E+00 + 0.0000E+00j, 0.1718E+00 + 0.0000E+00j,
                  0.2013E+00 + 0.0000E+00j, 0.2300E+00 + 0.0000E+00j,
                  0.2300E+00 + 0.0000E+00j, 0.2013E+00 + 0.0000E+00j,
                  0.1718E+00 + 0.0000E+00j, 0.1415E+00 + 0.0000E+00j,
                  0.1107E+00 + 0.0000E+00j, 0.7940E-01 + 0.0000E+00j,
                  0.4777E-01 + 0.0000E+00j, 0.1595E-01 + 0.0000E+00j,
                  0.0000E+00 + 0.0000E+00j, 0.0000E+00 + 0.0000E+00j,
                  0.0000E+00 + 0.0000E+00j, 0.0000E+00 + 0.0000E+00j,
                  0.0000E+00 + 0.0000E+00j, 0.0000E+00 + 0.0000E+00j,
                  0.0000E+00 + 0.0000E+00j, 0.0000E+00 + 0.0000E+00j,]
    assert_almost_equal(phi_golden, boundary_solution.phis, 0.0001)

    phi_golden = [0.1589E-01 + 0.1183E-03j,
                  0.4818E-01 + 0.4001E-04j,
                  0.4818E-01 + 0.4001E-04j,
                  0.1434E+00 - 0.2577E-03j,
                  0.6499E-01 - 0.1422E-04j,]
    assert_almost_equal(phi_golden, sample_solution.phis, 0.0001)


def test_internal_2d_2():
    # Test Problem 2
    # von Neumann boundary condition such that phi = sin(k/sqrt(2) * x) * sin(k/sqrt(2) * y)
    # Differentiate with respect to x and y to obtain outward normal:
    # dPhi/dX = k/sqrt(2) * cos(k/sqrt(2) * x) * sin(k/sqrt(2) * y)
    # dPhi/dY = k/sqrt(2) * sin(k/sqrt(2) * x) * cos(k/sqrt(2) * y)
    frequency = 400.0  # frequency [Hz]
    k, interior_points, interior_incident_phi, solver, boundary_incidence = initialize(frequency)

    boundary_condition = solver.neumann_boundary_condition()
    w = k / np.sqrt(2.0)
    for i in range(solver.centers.shape[0]):
        x = solver.centers[i, 0]
        y = solver.centers[i, 1]
        if (x < 1e-7):
            boundary_condition.f[i] = -w * np.cos(w * x) * np.sin(w * y)
        elif (x > 0.1 - 1e-7):
            boundary_condition.f[i] = w * np.cos(w * x) * np.sin(w * y)
        elif (y < 1e-7):
            boundary_condition.f[i] = -w * np.sin(w * x) * np.cos(w * y)
        else:
            boundary_condition.f[i] = w * np.sin(w * x) * np.cos(w * y)

    boundary_solution = solver.solve_boundary(k, boundary_condition, boundary_incidence)
    sample_solution = boundary_solution.solve_samples(interior_incident_phi, interior_points)

    phi_golden = [-.3647E-03 + 0.8013E-03j, -.3205E-03 + 0.8641E-03j,
                  -.2681E-03 + 0.9276E-03j, -.1668E-03 + 0.9850E-03j,
                  0.3219E-04 + 0.1031E-02j, 0.4063E-03 + 0.1060E-02j,
                  0.1118E-02 + 0.1066E-02j, 0.2859E-02 + 0.1050E-02j,
                  0.1937E-01 + 0.9291E-03j, 0.4881E-01 + 0.1150E-02j,
                  0.7902E-01 + 0.1426E-02j, 0.1092E+00 + 0.1723E-02j,
                  0.1388E+00 + 0.2028E-02j, 0.1677E+00 + 0.2333E-02j,
                  0.1954E+00 + 0.2627E-02j, 0.2205E+00 + 0.2877E-02j,
                  0.2205E+00 + 0.2877E-02j, 0.1954E+00 + 0.2627E-02j,
                  0.1677E+00 + 0.2333E-02j, 0.1388E+00 + 0.2028E-02j,
                  0.1092E+00 + 0.1723E-02j, 0.7902E-01 + 0.1426E-02j,
                  0.4881E-01 + 0.1150E-02j, 0.1937E-01 + 0.9291E-03j,
                  0.2859E-02 + 0.1050E-02j, 0.1118E-02 + 0.1066E-02j,
                  0.4063E-03 + 0.1060E-02j, 0.3219E-04 + 0.1031E-02j,
                  -.1668E-03 + 0.9850E-03j, -.2681E-03 + 0.9276E-03j,
                  -.3205E-03 + 0.8641E-03j, -.3647E-03 + 0.8013E-03j,]
    assert_almost_equal(phi_golden, boundary_solution.phis, 0.0001)

    phi_golden = [0.1559E-01 + 0.1166E-02j,
                  0.4805E-01 + 0.1369E-02j,
                  0.4805E-01 + 0.1369E-02j,
                  0.1398E+00 + 0.1898E-02j,
                  0.6392E-01 + 0.1475E-02j,]
    assert_almost_equal(phi_golden, sample_solution.phis, 0.0001)


def test_internal_2d_3():
    # Test Problem 3
    # The test problem computes the field produced by a unit source at
    # the point (0.5,0.25) within the square with a rigid boundary.
    # The rigid boundary implies the bondary condition v=0.
    # The test problem computes the field produced by a unit source at
    # the point (0.5,0.25) within the square with a rigid boundary.
    # The incident velocity potential is given by {\phi}_inc=i*h0(kr)/4
    # where r is the distance from the point (0.5,0.25)
    frequency = 400.0  # frequency [Hz]
    k, interior_points, interior_incident_phi, solver, boundary_incidence = initialize(frequency)

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
    sample_solution = boundary_solution.solve_samples(interior_incident_phi, interior_points)
    phi_golden = [-.1813E+01 + 0.3210E-03j, -.1827E+01 + 0.2720E-03j,
                  -.1855E+01 + 0.2822E-03j, -.1895E+01 + 0.3356E-03j,
                  -.1938E+01 + 0.4039E-03j, -.1976E+01 + 0.4646E-03j,
                  -.2004E+01 + 0.5079E-03j, -.2019E+01 + 0.5301E-03j,
                  -.2020E+01 + 0.4892E-03j, -.2018E+01 + 0.4470E-03j,
                  -.2016E+01 + 0.4173E-03j, -.2014E+01 + 0.4012E-03j,
                  -.2014E+01 + 0.4012E-03j, -.2016E+01 + 0.4173E-03j,
                  -.2018E+01 + 0.4470E-03j, -.2020E+01 + 0.4892E-03j,
                  -.2019E+01 + 0.5301E-03j, -.2004E+01 + 0.5079E-03j,
                  -.1976E+01 + 0.4646E-03j, -.1938E+01 + 0.4039E-03j,
                  -.1895E+01 + 0.3356E-03j, -.1855E+01 + 0.2822E-03j,
                  -.1827E+01 + 0.2720E-03j, -.1813E+01 + 0.3210E-03j,
                  -.1810E+01 + 0.4337E-03j, -.1782E+01 + 0.3972E-03j,
                  -.1731E+01 + 0.2555E-03j, -.1681E+01 + 0.3855E-04j,
                  -.1681E+01 + 0.3855E-04j, -.1731E+01 + 0.2555E-03j,
                  -.1782E+01 + 0.3972E-03j, -.1810E+01 + 0.4337E-03j,]
    assert_almost_equal(phi_golden, boundary_solution.phis, 0.001)

    phi_golden = [-.1777E+01 + 0.3587E-03j,
                  -.1777E+01 + 0.3587E-03j,
                  -.1984E+01 + 0.4169E-03j,
                  -.1984E+01 + 0.4169E-03j,
                  -.1845E+01 + 0.3876E-03j,]
    assert_almost_equal(phi_golden, sample_solution.phis, 0.001)
