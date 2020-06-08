import unittest
import numpy as np
from numpy.linalg import norm
from scipy.special import hankel1

import iops_pyx
import iops_cpp
import iops_sci


class TestComplexQuadGenerator(unittest.TestCase):
    def test_complex_quad_generator_01(self):
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 1.0], dtype=np.float32)

        def func(x):
            return 1.0

        result = iops_pyx.complex_quad_generator(func, a, b)
        self.assertAlmostEqual(result, np.sqrt(2.0), 6)


class TestCircularIntegratorPI(unittest.TestCase):
    def test_circular_integrator_01(self):
        circle = iops_pyx.CircularIntegratorPi(1)

        def func(x):
            return 1.0

        result = circle.integrate(func)
        self.assertAlmostEqual(result, np.pi)

    def test_circular_integrator_02(self):
        circle = iops_pyx.CircularIntegratorPi(2)

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
        result = iops_pyx.complex_quad_3d(func, a, b, c)
        self.assertAlmostEqual(result, 0.5)


class TestHankel(unittest.TestCase):
    def test_hankel_01(self):
        H1scipy = hankel1(0, 1.0)
        H1gsl = iops_cpp.hankel1_c(0, 1.0)
        self.assertAlmostEqual(H1scipy, H1gsl)

    def test_hankel_02(self):
        H1scipy = hankel1(0, 10.0)
        H1gsl = iops_cpp.hankel1_c(0, 10.0)
        self.assertAlmostEqual(H1scipy, H1gsl)

    def test_hankel_03(self):
        H1scipy = hankel1(1, 1.0)
        H1gsl = iops_cpp.hankel1_c(1, 1.0)
        self.assertAlmostEqual(H1scipy, H1gsl)

    def test_hankel_04(self):
        H1scipy = hankel1(1, 10.0)
        H1gsl = iops_cpp.hankel1_c(1, 10.0)
        self.assertAlmostEqual(H1scipy, H1gsl)


# -----------------------------------------------------------------------------
# 2D Integral Operators
# -----------------------------------------------------------------------------
class Test2D(unittest.TestCase):
    def setUp(self):
        self.a = np.array([0.5, 0.00], dtype=np.float32)
        self.b = np.array([0.0, 0.25], dtype=np.float32)
        self.p_off = np.array([1.0, 2.0], dtype=np.float32)
        self.p_on = (self.a + self.b) / 2.0; # center of mass for pOnElement
        self.n_p_off = np.array([-np.sqrt(0.5), -np.sqrt(0.5)], dtype=np.float32)
        ab = self.b - self.a
        self.n_p_on = np.array([-ab[1], ab[0]], dtype=np.float32)
        self.n_p_on = self.n_p_on / norm(self.n_p_on)


class TestComputeL_2D(Test2D):
    def test_compute_L_01(self):
        gld = -.62808617768766E-01+ 0.00000000000000E+00j
        k = 0.0
        p_on_element = False
        pyx = iops_pyx.l_2d(k, self.p_off, self.a, self.b, p_on_element)
        self.assertAlmostEqual(pyx, gld)
        cpp = iops_cpp.l_2d(k, self.p_off, self.a, self.b, p_on_element)
        self.assertAlmostEqual(cpp, gld)
        sci = iops_sci.l_2d(k, self.p_off, self.a, self.b, p_on_element)
        self.assertAlmostEqual(sci, gld)

    def test_compute_L_02(self):
        gld = -.38848700688676E-02+ 0.18666063352484E-01j
        k = 16.0
        p_on_element = False
        pyx = iops_pyx.l_2d(k, self.p_off, self.a, self.b, p_on_element)
        self.assertAlmostEqual(pyx, gld)
        cpp = iops_cpp.l_2d(k, self.p_off, self.a, self.b, p_on_element)
        self.assertAlmostEqual(cpp, gld)
        sci = iops_sci.l_2d(k, self.p_off, self.a, self.b, p_on_element)
        self.assertAlmostEqual(sci, gld)

    def test_compute_L_03(self):
        gld = 0.20238278599287E+00+ 0.00000000000000E+00j
        k = 0.0
        p_on_element = True
        pyx = iops_pyx.l_2d(k, self.p_on, self.a, self.b, p_on_element)
        self.assertAlmostEqual(pyx, gld)
        cpp = iops_cpp.l_2d(k, self.p_on, self.a, self.b, p_on_element)
        self.assertAlmostEqual(cpp, gld)
        sci = iops_sci.l_2d(k, self.p_on, self.a, self.b, p_on_element)
        self.assertAlmostEqual(sci, gld)

    def test_compute_L_04(self):
        gld = -.10438221373809E-01+ 0.26590088538927E-01j
        k = 16.0
        p_on_element = True
        pyx = iops_pyx.l_2d(k, self.p_on, self.a, self.b, p_on_element)
        self.assertAlmostEqual(pyx, gld)
        cpp = iops_cpp.l_2d(k, self.p_on, self.a, self.b, p_on_element)
        self.assertAlmostEqual(cpp, gld)
        sci = iops_sci.l_2d(k, self.p_on, self.a, self.b, p_on_element)
        self.assertAlmostEqual(sci, gld, 6)


class TestComputeM_2D(Test2D):
    def test_compute_M_01(self):
        gld = -.43635102946856E-01+ 0.00000000000000E+00j
        k = 0.0
        p_on_element = False
        pyx = iops_pyx.m_2d(k, self.p_off, self.a, self.b, p_on_element)
        self.assertAlmostEqual(pyx, gld)
        cpp = iops_cpp.m_2d(k, self.p_off, self.a, self.b, p_on_element)
        self.assertAlmostEqual(cpp, gld)
        sci = iops_sci.m_2d(k, self.p_off, self.a, self.b, p_on_element)
        self.assertAlmostEqual(sci, gld)

    def test_compute_M_02(self):
        gld = -0.29596284015305E+00 - 0.65862830497453E-01j
        k = 16.0
        p_on_element = False
        pyx = iops_pyx.m_2d(k, self.p_off, self.a, self.b, p_on_element)
        self.assertAlmostEqual(pyx, gld, 6)
        cpp = iops_cpp.m_2d(k, self.p_off, self.a, self.b, p_on_element)
        self.assertAlmostEqual(cpp, gld, 6)
        sci = iops_sci.m_2d(k, self.p_off, self.a, self.b, p_on_element)
        self.assertAlmostEqual(sci, gld, 6)


    def test_compute_M_03(self):
        gld = 0.00000000000000E+00+ 0.00000000000000E+00j
        k = 0.0
        p_on_element = True
        pyx = iops_pyx.m_2d(k, self.p_on, self.a, self.b, p_on_element)
        self.assertAlmostEqual(pyx, gld)
        cpp = iops_cpp.m_2d(k, self.p_on, self.a, self.b, p_on_element)
        self.assertAlmostEqual(cpp, gld)
        sci = iops_sci.m_2d(k, self.p_on, self.a, self.b, p_on_element)
        self.assertAlmostEqual(sci, gld, 5)

    def test_compute_M_04(self):
        gld = 0.00000000000000E+00+ 0.00000000000000E+00j
        k = 16.0
        p_on_element = True
        pyx = iops_pyx.m_2d(k, self.p_on, self.a, self.b, p_on_element)
        self.assertAlmostEqual(pyx, gld)
        cpp = iops_cpp.m_2d(k, self.p_on, self.a, self.b, p_on_element)
        self.assertAlmostEqual(cpp, gld)
        sci = iops_sci.m_2d(k, self.p_on, self.a, self.b, p_on_element)
        self.assertAlmostEqual(sci, gld,5)



class TestComputeMt_2D(Test2D):
    def test_compute_Mt_01(self):
        gld = 0.40260455651453E-01 + 0.00000000000000E+00j
        k = 0.0
        p_on_element = False
        pyx = iops_pyx.mt_2d(k, self.p_off, self.n_p_off, self.a, self.b, p_on_element)
        self.assertAlmostEqual(pyx, gld)
        cpp = iops_cpp.mt_2d(k, self.p_off, self.n_p_off, self.a, self.b, p_on_element)
        self.assertAlmostEqual(cpp, gld)
        sci = iops_sci.mt_2d(k, self.p_off, self.n_p_off, self.a, self.b, p_on_element)
        self.assertAlmostEqual(sci, gld)

    def test_compute_Mt_02(self):
        gld = 0.27354006901263E+00 + 0.59196279619442E-01j
        k = 16.0
        p_on_element = False
        pyx = iops_pyx.mt_2d(k, self.p_off, self.n_p_off, self.a, self.b, p_on_element)
        self.assertAlmostEqual(pyx, gld, 6)
        cpp = iops_cpp.mt_2d(k, self.p_off, self.n_p_off, self.a, self.b, p_on_element)
        self.assertAlmostEqual(cpp, gld, 6)
        sci = iops_sci.mt_2d(k, self.p_off, self.n_p_off, self.a, self.b, p_on_element)
        self.assertAlmostEqual(sci, gld, 6)

    def test_compute_Mt_03(self):
        gld = 0.00000000000000E+00 + 0.00000000000000E+00j
        k = 0.0
        p_on_element = True
        pyx = iops_pyx.mt_2d(k, self.p_on, self.n_p_on, self.a, self.b, p_on_element)
        self.assertAlmostEqual(pyx, gld)
        cpp = iops_cpp.mt_2d(k, self.p_on, self.n_p_on, self.a, self.b, p_on_element)
        self.assertAlmostEqual(cpp, gld)
        sci = iops_sci.mt_2d(k, self.p_on, self.n_p_on, self.a, self.b, p_on_element)
        self.assertAlmostEqual(sci, gld, 5)

    def test_compute_Mt_04(self):
        gld = 0.00000000000000E+00 + 0.00000000000000E+00j
        k = 16.0
        p_on_element = True
        pyx = iops_pyx.mt_2d(k, self.p_on, self.n_p_on, self.a, self.b, p_on_element)
        self.assertAlmostEqual(pyx, gld)
        cpp = iops_cpp.mt_2d(k, self.p_on, self.n_p_on, self.a, self.b, p_on_element)
        self.assertAlmostEqual(cpp, gld)
        sci = iops_sci.mt_2d(k, self.p_on, self.n_p_on, self.a, self.b, p_on_element)
        self.assertAlmostEqual(sci, gld, 5)


class TestComputeN_2D(Test2D):
    def test_compute_N_01(self):
        gld = -.18943306616838E-01 + 0.00000000000000E+00j
        k = 0.0
        p_on_element = False
        pyx = iops_pyx.n_2d(k, self.p_off, self.n_p_off, self.a, self.b, p_on_element)
        self.assertAlmostEqual(pyx, gld)
        cpp = iops_cpp.n_2d(k, self.p_off, self.n_p_off, self.a, self.b, p_on_element)
        self.assertAlmostEqual(cpp, gld)
        sci = iops_sci.n_2d(k, self.p_off, self.n_p_off, self.a, self.b, p_on_element)
        self.assertAlmostEqual(sci, gld)

    def test_compute_N_02(self):
        gld = -.99612499996911E+00 + 0.43379540259270E+01j
        k = 16.0
        p_on_element = False
        pyx = iops_pyx.n_2d(k, self.p_off, self.n_p_off, self.a, self.b, p_on_element)
        self.assertAlmostEqual(pyx, gld, 5)
        cpp = iops_cpp.n_2d(k, self.p_off, self.n_p_off, self.a, self.b, p_on_element)
        self.assertAlmostEqual(cpp, gld, 5)
        sci = iops_sci.n_2d(k, self.p_off, self.n_p_off, self.a, self.b, p_on_element)
        self.assertAlmostEqual(sci, gld, 5)

    def test_compute_N_03(self):
        gld = -.11388200377769E+01 + 0.00000000000000E+00j
        k = 0.0
        p_on_element = True
        pyx = iops_pyx.n_2d(k, self.p_on, self.n_p_on, self.a, self.b, p_on_element)
        self.assertAlmostEqual(pyx, gld)
        cpp = iops_cpp.n_2d(k, self.p_on, self.n_p_on, self.a, self.b, p_on_element)
        self.assertAlmostEqual(cpp, gld)
        sci = iops_sci.n_2d(k, self.p_on, self.n_p_off, self.a, self.b, p_on_element)
        self.assertAlmostEqual(sci, gld)

    def test_compute_N_04(self):
        gld = -.40622369223044E+00 + 0.85946767167784E+01j
        k = 16.0
        p_on_element = True
        pyx = iops_pyx.n_2d(k, self.p_on, self.n_p_on, self.a, self.b, p_on_element)
        self.assertAlmostEqual(pyx, gld, 4)
        cpp = iops_cpp.n_2d(k, self.p_on, self.n_p_on, self.a, self.b, p_on_element)
        self.assertAlmostEqual(cpp, gld, 5)
        sci = iops_sci.n_2d(k, self.p_on, self.n_p_on, self.a, self.b, p_on_element)
        self.assertAlmostEqual(sci, gld, 4)


# -----------------------------------------------------------------------------
# 3D Integral Operators
# -----------------------------------------------------------------------------
class Test3D(unittest.TestCase):
    def setUp(self):
        self.a = np.array([0.0, 0.00, 0.0], dtype=np.float32)
        self.b = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        self.c = np.array([0.0, 0.1, 0.0], dtype=np.float32)
        self.p_off = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        self.p_on = (self.a + self.b + self.c) / 3.0; # center of mass for pOnElement
        self.n_p_off = np.array([-np.sqrt(0.5), -np.sqrt(0.5), 0.0], dtype=np.float32)
        self.n_p_on = np.cross(self.b-self.a, self.c-self.a)
        self.n_p_on = self.n_p_on / norm(self.n_p_on)

class TestComputeL_3D(Test3D):
    def test_compute_L_01(self):
        k = 0.0
        p = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        a = np.array([0.0, 0.00, 0.0], dtype=np.float32)
        b = np.array([0.0, 0.10, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.00, 0.1], dtype=np.float32)
        pOnElement = False
        zP = iops_pyx.l_3d(k, self.p_off, a, self.b, c, pOnElement)
        zC = iops_cpp.l_3d(k, self.p_off, self.a, self.b, c, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_L_02(self):
        k = 10.0
        p = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        a = np.array([0.0, 0.00, 0.0], dtype=np.float32)
        b = np.array([0.0, 0.10, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.00, 0.1], dtype=np.float32)
        pOnElement = False
        zP = iops_pyx.l_3d(k, self.p_off, self.a, self.b, c, pOnElement)
        zC = iops_cpp.l_3d(k, self.p_off, self.a, self.b, c, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_L_03(self):
        k = 0.0
        p = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        a = np.array([0.0, 0.00, 0.0], dtype=np.float32)
        b = np.array([0.0, 0.10, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.00, 0.1], dtype=np.float32)
        pOnElement = True
        zP = iops_pyx.l_3d(k, self.p_on, self.a, self.b, c, pOnElement)
        zC = iops_cpp.l_3d(k, self.p_on, self.a, self.b, c, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_L_04(self):
        k = 10.0
        p = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        a = np.array([0.0, 0.00, 0.0], dtype=np.float32)
        b = np.array([0.0, 0.10, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.00, 0.1], dtype=np.float32)
        pOnElement = True
        zP = iops_pyx.l_3d(k, self.p_on, self.a, self.b, c, pOnElement)
        zC = iops_cpp.l_3d(k, self.p_on, self.a, self.b, c, pOnElement)
        self.assertAlmostEqual(zP, zC)


class TestComputeM_3D(Test3D):
    def test_compute_M_01(self):
        k = 0.0
        p = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        a = np.array([0.0, 0.00, 0.0], dtype=np.float32)
        b = np.array([0.0, 0.10, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.00, 0.1], dtype=np.float32)
        pOnElement = False
        zP = iops_pyx.m_3d(k, self.p_off, self.a, self.b, c, pOnElement)
        zC = iops_cpp.m_3d(k, self.p_off, self.a, self.b, c, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_M_02(self):
        k = 10.0
        p = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        a = np.array([0.0, 0.00, 0.0], dtype=np.float32)
        b = np.array([0.0, 0.10, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.00, 0.1], dtype=np.float32)
        pOnElement = False
        zP = iops_pyx.m_3d(k, self.p_off, self.a, self.b, c, pOnElement)
        zC = iops_cpp.m_3d(k, self.p_off, self.a, self.b, c, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_M_03(self):
        k = 0.0
        p = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        a = np.array([0.0, 0.00, 0.0], dtype=np.float32)
        b = np.array([0.0, 0.10, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.00, 0.1], dtype=np.float32)
        pOnElement = True
        zP = iops_pyx.m_3d(k, self.p_on, self.a, self.b, c, pOnElement)
        zC = iops_cpp.m_3d(k, self.p_on, self.a, self.b, c, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_M_04(self):
        k = 10.0
        p = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        a = np.array([0.0, 0.00, 0.0], dtype=np.float32)
        b = np.array([0.0, 0.10, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.00, 0.1], dtype=np.float32)
        pOnElement = True
        zP = iops_pyx.m_3d(k, self.p_on, self.a, self.b, c, pOnElement)
        zC = iops_cpp.m_3d(k, self.p_on, self.a, self.b, c, pOnElement)
        self.assertAlmostEqual(zP, zC)


class TestComputeMt_3D(Test3D):
    def test_compute_Mt_01(self):
        k = 0.0
        p = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5), 0.0], dtype=np.float32)
        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.1, 0.0], dtype=np.float32)
        pOnElement = False
        zP = iops_pyx.mt_3d(k, self.p_off, self.n_p_off, self.a, self.b, c, pOnElement)
        zC = iops_cpp.mt_3d(k, self.p_off, self.n_p_off, self.a, self.b, c, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_Mt_02(self):
        k = 10.0
        p = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5), 0.0], dtype=np.float32)
        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.1, 0.0], dtype=np.float32)
        pOnElement = False
        zP = iops_pyx.mt_3d(k, self.p_off, self.n_p_off, self.a, self.b, c, pOnElement)
        zC = iops_cpp.mt_3d(k, self.p_off, self.n_p_off, self.a, self.b, c, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_Mt_03(self):
        k = 0.0
        p = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5), 0.0], dtype=np.float32)
        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.1, 0.0], dtype=np.float32)
        pOnElement = True
        zP = iops_pyx.mt_3d(k, self.p_on, self.n_p_on, self.a, self.b, c, pOnElement)
        zC = iops_cpp.mt_3d(k, self.p_on, self.n_p_on, self.a, self.b, c, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_Mt_04(self):
        k = 10.0
        p = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5), 0.0], dtype=np.float32)
        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.1, 0.0], dtype=np.float32)
        pOnElement = True
        zP = iops_pyx.mt_3d(k, self.p_on, self.n_p_on, self.a, self.b, c, pOnElement)
        zC = iops_cpp.mt_3d(k, self.p_on, self.n_p_on, self.a, self.b, c, pOnElement)
        self.assertAlmostEqual(zP, zC)


class TestComputeN_3D(Test3D):
    def test_compute_N_01(self):
        k = 0.0
        p = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5), 0.0], dtype=np.float32)
        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.1, 0.0], dtype=np.float32)
        pOnElement = False
        zP = iops_pyx.n_3d(k, self.p_off, self.n_p_off, self.a, self.b, c, pOnElement)
        zC = iops_cpp.n_3d(k, self.p_off, self.n_p_off, self.a, self.b, c, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_N_02(self):
        k = 10.0
        p = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5), 0.0], dtype=np.float32)
        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.1, 0.0], dtype=np.float32)
        pOnElement = False
        zP = iops_pyx.n_3d(k, self.p_off, self.n_p_off, self.a, self.b, c, pOnElement)
        zC = iops_cpp.n_3d(k, self.p_off, self.n_p_off, self.a, self.b, c, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_N_03(self):
        k = 0.0
        p = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5), 0.0], dtype=np.float32)
        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.1, 0.0], dtype=np.float32)
        pOnElement = True
        zP = iops_pyx.n_3d(k, self.p_on, self.n_p_on, self.a, self.b, c, pOnElement)
        zC = iops_cpp.n_3d(k, self.p_on, self.n_p_on, self.a, self.b, c, pOnElement)
        self.assertAlmostEqual(zP, zC, 5)

    def test_compute_N_04(self):
        k = 10.0
        p = np.array([0.5, 0.75, 1.0], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5), 0.0], dtype=np.float32)
        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.1, 0.0], dtype=np.float32)
        pOnElement = True
        zP = iops_pyx.n_3d(k, self.p_on, self.n_p_on, self.a, self.b, c, pOnElement)
        zC = iops_cpp.n_3d(k, self.p_on, self.n_p_on, self.a, self.b, c, pOnElement)
        self.assertAlmostEqual(zP, zC, 4)


# -----------------------------------------------------------------------------
# Radial (RAD) Integral Operators
# -----------------------------------------------------------------------------
class TestRAD(unittest.TestCase):
    def setUp(self):
        self.a = np.array([0.5, 1.0], dtype=np.float32)
        self.b = np.array([1.0, 0.3], dtype=np.float32)
        self.p_off = np.array([0.3, 0.3], dtype=np.float32)
        self.p_on = (self.a + self.b) / 2.0; # center of mass for pOnElement
        self.n_p_off = np.array([-np.sqrt(0.5), -np.sqrt(0.5)], dtype=np.float32)
        d = self.b-self.a
        d = d / norm(d)
        self.n_p_on = np.array([d[1], -d[0]], dtype=np.float32)

class TestComputeL_RAD(TestRAD):
    def test_compute_L_01(self):
        k = 0.0
        p = np.array([0.3, 0.3], dtype=np.float32)
        a = np.array([0.5, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.3], dtype=np.float32)
        pOnElement = False
        zP = iops_pyx.l_rad(k, self.p_off, self.a, self.b, pOnElement)
        zC = iops_cpp.l_rad(k, self.p_off, self.a, self.b, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_L_02(self):
        k = 10.0
        p = np.array([0.3, 0.3], dtype=np.float32)
        a = np.array([0.5, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.3], dtype=np.float32)
        pOnElement = False
        zP = iops_pyx.l_rad(k, self.p_off, self.a, self.b, pOnElement)
        zC = iops_cpp.l_rad(k, self.p_off, self.a, self.b, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_L_03(self):
        k = 0.0
        p = np.array([0.75, 0.75], dtype=np.float32)
        a = np.array([0.5, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.5], dtype=np.float32)
        pOnElement = True
        zP = iops_pyx.l_rad(k, self.p_on, self.a, self.b, pOnElement)
        zC = iops_cpp.l_rad(k, self.p_on, self.a, self.b, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_L_04(self):
        k = 10.0
        p = np.array([0.75, 0.75], dtype=np.float32)
        a = np.array([0.5, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.5], dtype=np.float32)
        pOnElement = True
        zP = iops_pyx.l_rad(k, self.p_on, self.a, self.b, pOnElement)
        zC = iops_cpp.l_rad(k, self.p_on, self.a, self.b, pOnElement)
        self.assertAlmostEqual(zP, zC)


class TestComputeM_RAD(TestRAD):
    def test_compute_M_01(self):
        k = 0.0
        p = np.array([0.3, 0.3], dtype=np.float32)
        a = np.array([0.5, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.3], dtype=np.float32)
        pOnElement = False
        zP = iops_pyx.m_rad(k, self.p_off, self.a, self.b, pOnElement)
        zC = iops_cpp.m_rad(k, self.p_off, self.a, self.b, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_M_02(self):
        k = 10.0
        p = np.array([0.3, 0.3], dtype=np.float32)
        a = np.array([0.5, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.3], dtype=np.float32)
        pOnElement = False
        zP = iops_pyx.m_rad(k, self.p_off, self.a, self.b, pOnElement)
        zC = iops_cpp.m_rad(k, self.p_off, self.a, self.b, pOnElement)
        self.assertAlmostEqual(zP, zC, 6)

    def test_compute_M_03(self):
        k = 0.0
        p = np.array([0.75, 0.75], dtype=np.float32)
        a = np.array([0.5, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.5], dtype=np.float32)
        pOnElement = True
        zP = iops_pyx.m_rad(k, self.p_on, self.a, self.b, pOnElement)
        zC = iops_cpp.m_rad(k, self.p_on, self.a, self.b, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_M_04(self):
        k = 10.0
        p = np.array([0.75, 0.75], dtype=np.float32)
        a = np.array([0.5, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.5], dtype=np.float32)
        pOnElement = True
        zP = iops_pyx.m_rad(k, self.p_on, self.a, self.b, pOnElement)
        zC = iops_cpp.m_rad(k, self.p_on, self.a, self.b, pOnElement)
        self.assertAlmostEqual(zP, zC, 6)


class TestComputeMt_RAD(TestRAD):
    def test_compute_Mt_01(self):
        k = 0.0
        p = np.array([0.3, 0.3], dtype=np.float32)
        a = np.array([0.5, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.3], dtype=np.float32)
        normal_p = np.array([np.sqrt(0.5), np.sqrt(0.5)], dtype=np.float32)
        pOnElement = False
        zP = iops_pyx.mt_rad(k, self.p_off, self.n_p_off, self.a, self.b, pOnElement)
        zC = iops_cpp.mt_rad(k, self.p_off, self.n_p_off, self.a, self.b, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_Mt_02(self):
        k = 10.0
        p = np.array([0.3, 0.3], dtype=np.float32)
        a = np.array([0.5, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.3], dtype=np.float32)
        normal_p = np.array([np.sqrt(0.5), np.sqrt(0.5)], dtype=np.float32)
        pOnElement = False
        zP = iops_pyx.mt_rad(k, self.p_off, self.n_p_off, self.a, self.b, pOnElement)
        zC = iops_cpp.mt_rad(k, self.p_off, self.n_p_off, self.a, self.b, pOnElement)
        self.assertAlmostEqual(zP, zC, 6)

    def test_compute_Mt_03(self):
        k = 0.0
        p = np.array([0.75, 0.75], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5)], dtype=np.float32)
        a = np.array([0.5, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.5], dtype=np.float32)
        pOnElement = True
        zP = iops_pyx.mt_rad(k, self.p_on, self.n_p_on, self.a, self.b, pOnElement)
        zC = iops_cpp.mt_rad(k, self.p_on, self.n_p_on, self.a, self.b, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_Mt_04(self):
        k = 10.0
        p = np.array([0.75, 0.75], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5)], dtype=np.float32)
        a = np.array([0.5, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.5], dtype=np.float32)
        pOnElement = True
        zP = iops_pyx.mt_rad(k, self.p_on, self.n_p_on, self.a, self.b, pOnElement)
        zC = iops_cpp.mt_rad(k, self.p_on, self.n_p_on, self.a, self.b, pOnElement)
        self.assertAlmostEqual(zP, zC, 6)


class TestComputeN_RAD(TestRAD):
    def test_compute_N_01(self):
        k = 0.0
        p = np.array([0.3, 0.3], dtype=np.float32)
        a = np.array([0.5, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.3], dtype=np.float32)
        normal_p = np.array([np.sqrt(0.5), np.sqrt(0.5)], dtype=np.float32)
        pOnElement = False
        zP = iops_pyx.n_rad(k, self.p_off, self.n_p_off, self.a, self.b, pOnElement)
        zC = iops_cpp.n_rad(k, self.p_off, self.n_p_off, self.a, self.b, pOnElement)
        self.assertAlmostEqual(zP, zC)

    def test_compute_N_02(self):
        k = 10.0
        p = np.array([0.3, 0.3], dtype=np.float32)
        a = np.array([0.5, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.3], dtype=np.float32)
        normal_p = np.array([np.sqrt(0.5), np.sqrt(0.5)], dtype=np.float32)
        pOnElement = False
        zP = iops_pyx.n_rad(k, self.p_off, self.n_p_off, self.a, self.b, pOnElement)
        zC = iops_cpp.n_rad(k, self.p_off, self.n_p_off, self.a, self.b, pOnElement)
        self.assertAlmostEqual(zP, zC, 6)

    def test_compute_N_03(self):
        k = 0.0
        p = np.array([0.75, 0.75], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5)], dtype=np.float32)
        a = np.array([0.5, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.5], dtype=np.float32)
        pOnElement = True
        zP = iops_pyx.n_rad(k, self.p_on, self.n_p_on, self.a, self.b, pOnElement)
        zC = iops_cpp.n_rad(k, self.p_on, self.n_p_on, self.a, self.b, pOnElement)
        self.assertAlmostEqual(zP, zC, 6)

    def test_compute_N_04(self):
        k = 10.0
        p = np.array([0.75, 0.75], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5)], dtype=np.float32)
        a = np.array([0.5, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.5], dtype=np.float32)
        pOnElement = True
        zP = iops_pyx.n_rad(k, self.p_on, self.n_p_on, self.a, self.b, pOnElement)
        zC = iops_cpp.n_rad(k, self.p_on, self.n_p_on, self.a, self.b, pOnElement)
        self.assertAlmostEqual(zP, zC, 4)


if __name__ == "__main__":
    unittest.main()
