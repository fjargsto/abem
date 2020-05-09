import unittest
import numpy as np
from scipy.special import hankel1

from abem.helmholtz_integrals_2d import l_2d_p, m_2d_p, mt_2d_p, n_2d_p
from intops import *
from abem.helmholtz_integrals_3d import l_3d_p, m_3d_p, mt_3d_p, n_3d_p, complex_quad
from abem.helmholtz_integrals_rad import l_rad_p, m_rad_p, mt_rad_p, n_rad_p, complex_quad_generator, CircularIntegratorPi


class TestComplexQuadGenerator(unittest.TestCase):
    def test_complex_quad_generator_01(self):
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 1.0], dtype=np.float32)

        def func(x):
            return 1.0

        result = complex_quad_generator(func, a, b)
        self.assertAlmostEqual(result, np.sqrt(2.0), 6)


class TestCircularIntegratorPI(unittest.TestCase):
    def test_circular_integrator_01(self):
        circle = CircularIntegratorPi(1)

        def func(x):
            return 1.0

        result = circle.integrate(func)
        self.assertAlmostEqual(result, np.pi)

    def test_circular_integrator_02(self):
        circle = CircularIntegratorPi(2)

        def func(x):
            return 1.0

        result = circle.integrate(func)
        self.assertAlmostEqual(result, np.pi)


class TestTriangleIntegrator(unittest.TestCase):
    def test_complex_quad(self):
        def func(x):
            return 1.0

        a = np.array([0, 0, 0], dtype=np.float32)
        b = np.array([0, 1, 0], dtype=np.float32)
        c = np.array([0, 0, 1], dtype=np.float32)
        result = complex_quad(func, a, b, c)
        self.assertAlmostEqual(result, 0.5)


class TestHankel(unittest.TestCase):
    def test_hankel_01(self):
        H1scipy = hankel1(0, 1.0)
        H1gsl = hankel1_c(0, 1.0)
        self.assertAlmostEqual(H1scipy, H1gsl)

    def test_hankel_02(self):
        H1scipy = hankel1(0, 10.0)
        H1gsl = hankel1_c(0, 10.0)
        self.assertAlmostEqual(H1scipy, H1gsl)

    def test_hankel_03(self):
        H1scipy = hankel1(1, 1.0)
        H1gsl = hankel1_c(1, 1.0)
        self.assertAlmostEqual(H1scipy, H1gsl)

    def test_hankel_04(self):
        H1scipy = hankel1(1, 10.0)
        H1gsl = hankel1_c(1, 10.0)
        self.assertAlmostEqual(H1scipy, H1gsl)


# -----------------------------------------------------------------------------
# 2D Integral Operators
# -----------------------------------------------------------------------------
class TestComputeL_2D(unittest.TestCase):
    def test_compute_L_01(self):
        k = 0.0
        p = np.array([0.5, 0.75], dtype=np.float32)
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = False
        zP = l_2d_p(k, p, a, b, pOnElement)
        zC = l_2d_c(k, p, a, b, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_L_02(self):
        k = 10.0
        p = np.array([0.5, 0.75], dtype=np.float32)
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = False
        zP = l_2d_p(k, p, a, b, pOnElement)
        zC = l_2d_c(k, p, a, b, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_L_03(self):
        k = 0.0
        p = np.array([0.0, 0.05], dtype=np.float32)
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = True
        zP = l_2d_p(k, p, a, b, pOnElement)
        zC = l_2d_c(k, p, a, b, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_L_04(self):
        k = 10.0
        p = np.array([0.0, 0.05], dtype=np.float32)
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = True
        zP = l_2d_p(k, p, a, b, pOnElement)
        zC = l_2d_c(k, p, a, b, pOnElement)
        self.assertAlmostEqual(zP, zC)


class TestComputeM_2D(unittest.TestCase):
    def test_compute_M_01(self):
        k = 0.0
        p = np.array([0.5, 0.75], dtype=np.float32)
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = False
        zP = m_2d_p(k, p, a, b, pOnElement)
        zC = m_2d_c(k, p, a, b, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_M_02(self):
        k = 10.0
        p = np.array([0.5, 0.75], dtype=np.float32)
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = False
        zP = m_2d_p(k, p, a, b, pOnElement)
        zC = m_2d_c(k, p, a, b, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_M_03(self):
        k = 0.0
        p = np.array([0.0, 0.05], dtype=np.float32)
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = True
        zP = m_2d_p(k, p, a, b, pOnElement)
        zC = m_2d_c(k, p, a, b, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_M_04(self):
        k = 10.0
        p = np.array([0.0, 0.05], dtype=np.float32)
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = True
        zP = m_2d_p(k, p, a, b, pOnElement)
        zC = m_2d_c(k, p, a, b, pOnElement)
        self.assertAlmostEqual(zP, zC)


class TestComputeMt_2D(unittest.TestCase):
    def test_compute_Mt_01(self):
        k = 0.0
        p = np.array([0.5, 0.75], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5)], dtype=np.float32)
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = False
        zP = mt_2d_p(k, p, normal_p, a, b, pOnElement)
        zC = mt_2d_c(k, p, normal_p, a, b, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_Mt_02(self):
        k = 10.0
        p = np.array([0.5, 0.75], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5)], dtype=np.float32)
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = False
        zP = mt_2d_p(k, p, normal_p, a, b, pOnElement)
        zC = mt_2d_c(k, p, normal_p, a, b, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_Mt_03(self):
        k = 0.0
        p = np.array([0.0, 0.05], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5)], dtype=np.float32)
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = True
        zP = mt_2d_p(k, p, normal_p, a, b, pOnElement)
        zC = mt_2d_c(k, p, normal_p, a, b, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_Mt_04(self):
        k = 10.0
        p = np.array([0.0, 0.05], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5)], dtype=np.float32)
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = True
        zP = mt_2d_p(k, p, normal_p, a, b, pOnElement)
        zC = mt_2d_c(k, p, normal_p, a, b, pOnElement)
        self.assertAlmostEqual(zP, zC)


class TestComputeN_2D(unittest.TestCase):
    def test_compute_N_01(self):
        k = 0.0
        p = np.array([0.5, 0.75], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5)], dtype=np.float32)
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = False
        zP = n_2d_p(k, p, normal_p, a, b, pOnElement)
        zC = n_2d_c(k, p, normal_p, a, b, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_N_02(self):
        k = 10.0
        p = np.array([0.5, 0.75], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5)], dtype=np.float32)
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = False
        zP = n_2d_p(k, p, normal_p, a, b, pOnElement)
        zC = n_2d_c(k, p, normal_p, a, b, pOnElement)
        self.assertAlmostEqual(zP, zC, 6)

    def test_compute_N_03(self):
        k = 0.0
        p = np.array([0.0, 0.05], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5)], dtype=np.float32)
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = True
        zP = n_2d_p(k, p, normal_p, a, b, pOnElement)
        zC = n_2d_c(k, p, normal_p, a, b, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_N_04(self):
        k = 10.0
        p = np.array([0.0, 0.05], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5)], dtype=np.float32)
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = True
        zP = n_2d_p(k, p, normal_p, a, b, pOnElement)
        zC = n_2d_c(k, p, normal_p, a, b, pOnElement)
        self.assertAlmostEqual(zP, zC, 3)


# -----------------------------------------------------------------------------
# 3D Integral Operators
# -----------------------------------------------------------------------------
class TestComputeL_3D(unittest.TestCase):
    def test_compute_L_01(self):
        k = 0.0
        p = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        a = np.array([0.0, 0.00, 0.0], dtype=np.float32)
        b = np.array([0.0, 0.10, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.00, 0.1], dtype=np.float32)
        pOnElement = False
        zP = l_3d_p(k, p, a, b, c, pOnElement)
        zC = l_3d_c(k, p, a, b, c, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_L_02(self):
        k = 10.0
        p = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        a = np.array([0.0, 0.00, 0.0], dtype=np.float32)
        b = np.array([0.0, 0.10, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.00, 0.1], dtype=np.float32)
        pOnElement = False
        zP = l_3d_p(k, p, a, b, c, pOnElement)
        zC = l_3d_c(k, p, a, b, c, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_L_03(self):
        k = 0.0
        p = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        a = np.array([0.0, 0.00, 0.0], dtype=np.float32)
        b = np.array([0.0, 0.10, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.00, 0.1], dtype=np.float32)
        pOnElement = True
        zP = l_3d_p(k, p, a, b, c, pOnElement)
        zC = l_3d_c(k, p, a, b, c, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_L_04(self):
        k = 10.0
        p = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        a = np.array([0.0, 0.00, 0.0], dtype=np.float32)
        b = np.array([0.0, 0.10, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.00, 0.1], dtype=np.float32)
        pOnElement = True
        zP = l_3d_p(k, p, a, b, c, pOnElement)
        zC = l_3d_c(k, p, a, b, c, pOnElement)
        self.assertAlmostEqual(zP, zC)


class TestComputeM_3D(unittest.TestCase):
    def test_compute_M_01(self):
        k = 0.0
        p = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        a = np.array([0.0, 0.00, 0.0], dtype=np.float32)
        b = np.array([0.0, 0.10, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.00, 0.1], dtype=np.float32)
        pOnElement = False
        zP = m_3d_p(k, p, a, b, c, pOnElement)
        zC = m_3d_c(k, p, a, b, c, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_M_02(self):
        k = 10.0
        p = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        a = np.array([0.0, 0.00, 0.0], dtype=np.float32)
        b = np.array([0.0, 0.10, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.00, 0.1], dtype=np.float32)
        pOnElement = False
        zP = m_3d_p(k, p, a, b, c, pOnElement)
        zC = m_3d_c(k, p, a, b, c, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_M_03(self):
        k = 0.0
        p = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        a = np.array([0.0, 0.00, 0.0], dtype=np.float32)
        b = np.array([0.0, 0.10, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.00, 0.1], dtype=np.float32)
        pOnElement = True
        zP = m_3d_p(k, p, a, b, c, pOnElement)
        zC = m_3d_c(k, p, a, b, c, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_M_04(self):
        k = 10.0
        p = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        a = np.array([0.0, 0.00, 0.0], dtype=np.float32)
        b = np.array([0.0, 0.10, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.00, 0.1], dtype=np.float32)
        pOnElement = True
        zP = m_3d_p(k, p, a, b, c, pOnElement)
        zC = m_3d_c(k, p, a, b, c, pOnElement)
        self.assertAlmostEqual(zP, zC)


class TestComputeMt_3D(unittest.TestCase):
    def test_compute_Mt_01(self):
        k = 0.0
        p = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5), 0.0], dtype=np.float32)
        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.1, 0.0], dtype=np.float32)
        pOnElement = False
        zP = mt_3d_p(k, p, normal_p, a, b, c, pOnElement)
        zC = mt_3d_c(k, p, normal_p, a, b, c, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_Mt_02(self):
        k = 10.0
        p = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5), 0.0], dtype=np.float32)
        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.1, 0.0], dtype=np.float32)
        pOnElement = False
        zP = mt_3d_p(k, p, normal_p, a, b, c, pOnElement)
        zC = mt_3d_c(k, p, normal_p, a, b, c, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_Mt_03(self):
        k = 0.0
        p = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5), 0.0], dtype=np.float32)
        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.1, 0.0], dtype=np.float32)
        pOnElement = True
        zP = mt_3d_p(k, p, normal_p, a, b, c, pOnElement)
        zC = mt_3d_c(k, p, normal_p, a, b, c, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_Mt_04(self):
        k = 10.0
        p = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5), 0.0], dtype=np.float32)
        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.1, 0.0], dtype=np.float32)
        pOnElement = True
        zP = mt_3d_p(k, p, normal_p, a, b, c, pOnElement)
        zC = mt_3d_c(k, p, normal_p, a, b, c, pOnElement)
        self.assertAlmostEqual(zP, zC)


class TestComputeN_3D(unittest.TestCase):
    def test_compute_N_01(self):
        k = 0.0
        p = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5), 0.0], dtype=np.float32)
        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.1, 0.0], dtype=np.float32)
        pOnElement = False
        zP = n_3d_p(k, p, normal_p, a, b, c, pOnElement)
        zC = n_3d_c(k, p, normal_p, a, b, c, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_N_02(self):
        k = 10.0
        p = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5), 0.0], dtype=np.float32)
        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.1, 0.0], dtype=np.float32)
        pOnElement = False
        zP = n_3d_p(k, p, normal_p, a, b, c, pOnElement)
        zC = n_3d_c(k, p, normal_p, a, b, c, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_N_03(self):
        k = 0.0
        p = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5), 0.0], dtype=np.float32)
        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.1, 0.0], dtype=np.float32)
        pOnElement = True
        zP = n_3d_p(k, p, normal_p, a, b, c, pOnElement)
        zC = n_3d_c(k, p, normal_p, a, b, c, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_N_04(self):
        k = 10.0
        p = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5), 0.0], dtype=np.float32)
        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.1, 0.0], dtype=np.float32)
        pOnElement = True
        zP = n_3d_p(k, p, normal_p, a, b, c, pOnElement)
        zC = n_3d_c(k, p, normal_p, a, b, c, pOnElement)
        self.assertAlmostEqual(zP, zC, 5)


# -----------------------------------------------------------------------------
# Radial (RAD) Integral Operators
# -----------------------------------------------------------------------------
class TestComputeL_RAD(unittest.TestCase):
    def test_compute_L_01(self):
        k = 0.0
        p = np.array([0.3, 0.3], dtype=np.float32)
        a = np.array([0.5, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.3], dtype=np.float32)
        pOnElement = False
        zP = l_rad_p(k, p, a, b, pOnElement)
        zC = l_rad_c(k, p, a, b, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_L_02(self):
        k = 10.0
        p = np.array([0.3, 0.3], dtype=np.float32)
        a = np.array([0.5, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.3], dtype=np.float32)
        pOnElement = False
        zP = l_rad_p(k, p, a, b, pOnElement)
        zC = l_rad_c(k, p, a, b, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_L_03(self):
        k = 0.0
        p = np.array([0.75, 0.75], dtype=np.float32)
        a = np.array([0.5, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.5], dtype=np.float32)
        pOnElement = True
        zP = l_rad_p(k, p, a, b, pOnElement)
        zC = l_rad_c(k, p, a, b, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_L_04(self):
        k = 10.0
        p = np.array([0.75, 0.75], dtype=np.float32)
        a = np.array([0.5, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.5], dtype=np.float32)
        pOnElement = True
        zP = l_rad_p(k, p, a, b, pOnElement)
        zC = l_rad_c(k, p, a, b, pOnElement)
        self.assertAlmostEqual(zP, zC)


class TestComputeM_RAD(unittest.TestCase):
    def test_compute_M_01(self):
        k = 0.0
        p = np.array([0.3, 0.3], dtype=np.float32)
        a = np.array([0.5, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.3], dtype=np.float32)
        pOnElement = False
        zP = m_rad_p(k, p, a, b, pOnElement)
        zC = m_rad_c(k, p, a, b, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_M_02(self):
        k = 10.0
        p = np.array([0.3, 0.3], dtype=np.float32)
        a = np.array([0.5, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.3], dtype=np.float32)
        pOnElement = False
        zP = m_rad_p(k, p, a, b, pOnElement)
        zC = m_rad_c(k, p, a, b, pOnElement)
        self.assertAlmostEqual(zP, zC, 6)

    def test_compute_M_03(self):
        k = 0.0
        p = np.array([0.75, 0.75], dtype=np.float32)
        a = np.array([0.5, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.5], dtype=np.float32)
        pOnElement = True
        zP = m_rad_p(k, p, a, b, pOnElement)
        zC = m_rad_c(k, p, a, b, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_M_04(self):
        k = 10.0
        p = np.array([0.75, 0.75], dtype=np.float32)
        a = np.array([0.5, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.5], dtype=np.float32)
        pOnElement = True
        zP = m_rad_p(k, p, a, b, pOnElement)
        zC = m_rad_c(k, p, a, b, pOnElement)
        self.assertAlmostEqual(zP, zC, 6)


class TestComputeMt_RAD(unittest.TestCase):
    def test_compute_Mt_01(self):
        k = 0.0
        p = np.array([0.3, 0.3], dtype=np.float32)
        a = np.array([0.5, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.3], dtype=np.float32)
        normal_p = np.array([np.sqrt(0.5), np.sqrt(0.5)], dtype=np.float32)
        pOnElement = False
        zP = mt_rad_p(k, p, normal_p, a, b, pOnElement)
        zC = mt_rad_c(k, p, normal_p, a, b, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_Mt_02(self):
        k = 10.0
        p = np.array([0.3, 0.3], dtype=np.float32)
        a = np.array([0.5, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.3], dtype=np.float32)
        normal_p = np.array([np.sqrt(0.5), np.sqrt(0.5)], dtype=np.float32)
        pOnElement = False
        zP = mt_rad_p(k, p, normal_p, a, b, pOnElement)
        zC = mt_rad_c(k, p, normal_p, a, b, pOnElement)
        self.assertAlmostEqual(zP, zC, 6)

    def test_compute_Mt_03(self):
        k = 0.0
        p = np.array([0.75, 0.75], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5)], dtype=np.float32)
        a = np.array([0.5, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.5], dtype=np.float32)
        pOnElement = True
        zP = mt_rad_p(k, p, normal_p, a, b, pOnElement)
        zC = mt_rad_c(k, p, normal_p, a, b, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_Mt_04(self):
        k = 10.0
        p = np.array([0.75, 0.75], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5)], dtype=np.float32)
        a = np.array([0.5, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.5], dtype=np.float32)
        pOnElement = True
        zP = mt_rad_p(k, p, normal_p, a, b, pOnElement)
        zC = mt_rad_c(k, p, normal_p, a, b, pOnElement)
        self.assertAlmostEqual(zP, zC)


class TestComputeN_RAD(unittest.TestCase):
    def test_compute_N_01(self):
        k = 0.0
        p = np.array([0.3, 0.3], dtype=np.float32)
        a = np.array([0.5, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.3], dtype=np.float32)
        normal_p = np.array([np.sqrt(0.5), np.sqrt(0.5)], dtype=np.float32)
        pOnElement = False
        zP = n_rad_p(k, p, normal_p, a, b, pOnElement)
        zC = n_rad_c(k, p, normal_p, a, b, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_N_02(self):
        k = 10.0
        p = np.array([0.3, 0.3], dtype=np.float32)
        a = np.array([0.5, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.3], dtype=np.float32)
        normal_p = np.array([np.sqrt(0.5), np.sqrt(0.5)], dtype=np.float32)
        pOnElement = False
        zP = n_rad_p(k, p, normal_p, a, b, pOnElement)
        zC = n_rad_c(k, p, normal_p, a, b, pOnElement)
        self.assertAlmostEqual(zP, zC, 6)

    def test_compute_N_03(self):
        k = 0.0
        p = np.array([0.75, 0.75], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5)], dtype=np.float32)
        a = np.array([0.5, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.5], dtype=np.float32)
        pOnElement = True
        zP = n_rad_p(k, p, normal_p, a, b, pOnElement)
        zC = n_rad_c(k, p, normal_p, a, b, pOnElement)
        self.assertAlmostEqual(zP, zC, 6)

    def test_compute_N_04(self):
        k = 10.0
        p = np.array([0.75, 0.75], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5)], dtype=np.float32)
        a = np.array([0.5, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.5], dtype=np.float32)
        pOnElement = True
        zP = n_rad_p(k, p, normal_p, a, b, pOnElement)
        zC = n_rad_c(k, p, normal_p, a, b, pOnElement)
        self.assertAlmostEqual(zP, zC, 4)


if __name__ == "__main__":
    unittest.main()
