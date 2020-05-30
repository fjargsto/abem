import numpy as np
cimport numpy as np
from numpy.linalg import norm
from scipy.special import hankel1

cdef extern from "stdbool.h":
    ctypedef bint bool


cpdef normal_2d(np.ndarray a, np.ndarray b):
    diff = a - b
    length = norm(diff)
    return np.array([diff[1]/length, -diff[0]/length])

cpdef normal_3d(np.ndarray a, np.ndarray b, np.ndarray c):
    ab = b - a
    ac = c - a
    normal = np.cross(ab, ac)
    normal /= norm(normal)
    return normal

cpdef complex_quad_2d(func, np.ndarray start, np.ndarray end):
    samples = np.array(
        [
            [0.980144928249, 5.061426814519e-02],
            [0.898333238707, 0.111190517227],
            [0.762766204958, 0.156853322939],
            [0.591717321248, 0.181341891689],
            [0.408282678752, 0.181341891689],
            [0.237233795042, 0.156853322939],
            [0.101666761293, 0.111190517227],
            [1.985507175123e-02, 5.061426814519e-02],
        ],
        dtype=np.float32,
    )
    vec = end - start
    sum = 0.0
    for n in range(samples.shape[0]):
        x = start + samples[n, 0] * vec
        sum += samples[n, 1] * func(x)
    return sum * norm(vec)


# -----------------------------------------------------------------------------
# 2D
# -----------------------------------------------------------------------------
def l_2d(float k, np.ndarray p, np.ndarray qa, np.ndarray qb, bool p_on_element):
    qab = qb - qa
    if p_on_element:
        if k == 0.0:
            ra = norm(p - qa)
            rb = norm(p - qb)
            re = norm(qab)
            result = 0.5 / np.pi * (re - (ra * np.log(ra) + rb * np.log(rb)))
        else:

            def func(x):
                R = norm(p - x)
                return 0.5 / np.pi * np.log(R) + 0.25j * hankel1(0, k * R)

            result = (
                complex_quad_2d(func, qa, p)
                + complex_quad_2d(func, p, qb)
                + l_2d(0.0, p, qa, qb, True)
            )
    else:
        if k == 0.0:
            result = -0.5 / np.pi * complex_quad_2d(lambda q: np.log(norm(p - q)), qa, qb)
        else:
            result =  0.25j * complex_quad_2d(lambda q: hankel1(0, k * norm(p - q)), qa, qb)

    return result


def m_2d(float k, np.ndarray p, np.ndarray qa, np.ndarray qb, bool p_on_element):
    vecq = normal_2d(qa, qb)
    if p_on_element:
        result = 0.0
    else:
        if k == 0.0:

            def func(x):
                r = p - x
                return np.dot(r, vecq) / np.dot(r, r)

            result = -0.5 / np.pi * complex_quad_2d(func, qa, qb)
        else:

            def func(x):
                r = p - x
                R = norm(r)
                return hankel1(1, k * R) * np.dot(r, vecq) / R

            result = 0.25j * k * complex_quad_2d(func, qa, qb)

    return result


def mt_2d(float k, np.ndarray p, np.ndarray vecp, np.ndarray qa, np.ndarray qb, bool p_on_element):
    if p_on_element:
        return 0.0
    else:
        if k == 0.0:

            def func(x):
                r = p - x
                return np.dot(r, vecp) / np.dot(r, r)

            return -0.5 / np.pi * complex_quad_2d(func, qa, qb)
        else:

            def func(x):
                r = p - x
                R = norm(r)
                return hankel1(1, k * R) * np.dot(r, vecp) / R

            return -0.25j * k * complex_quad_2d(func, qa, qb)


def n_2d(float k, np.ndarray p, np.ndarray vecp, np.ndarray qa, np.ndarray qb, bool p_on_element):
    qab = qb - qa
    if p_on_element:
        ra = norm(p - qa)
        rb = norm(p - qb)
        re = norm(qab)
        if k == 0.0:
            result = -(1.0 / ra + 1.0 / rb) / (2.0 * np.pi)
            return result
        else:
            vecq = normal_2d(qa, qb)
            k_sqr = k * k

            def func(x):
                r = p - x
                R2 = np.dot(r, r)
                R = np.sqrt(R2)
                drdudrdn = -np.dot(r, vecq) * np.dot(r, vecp) / R2
                dpnu = np.dot(vecp, vecq)
                c1 = 0.25j * k / R * hankel1(1, k * R) - 0.5 / (np.pi * R2)
                c2 = (
                    0.50j * k / R * hankel1(1, k * R)
                    - 0.25j * k_sqr * hankel1(0, k * R)
                    - 1.0 / (np.pi * R2)
                )
                c3 = -0.25 * k_sqr * np.log(R) / np.pi
                return c1 * dpnu + c2 * drdudrdn + c3

            n_0 = n_2d(0.0, p, vecp, qa, qb, True)
            l_0 = - 0.5 * k_sqr * l_2d(0.0, p, qa, qb, True)
            integral = complex_quad_2d(func, qa, p) + complex_quad_2d(func, p, qb)

            return integral + n_0 + l_0
    else:
        vecq = normal_2d(qa, qb)
        un = np.dot(vecp, vecq)
        if k == 0.0:

            def func(x):
                r = p - x
                R2 = np.dot(r, r)
                drdudrdn = -np.dot(r, vecq) * np.dot(r, vecp) / R2
                return (un + 2.0 * drdudrdn) / R2

            return 0.5 / np.pi * complex_quad_2d(func, qa, qb)
        else:

            def func(x):
                r = p - x
                drdudrdn = -np.dot(r, vecq) * np.dot(r, vecp) / np.dot(r, r)
                R = norm(r)
                return (
                    hankel1(1, k * R) / R * (un + 2.0 * drdudrdn)
                    - k * hankel1(0, k * R) * drdudrdn
                )

            return 0.25j * k * complex_quad_2d(func, qa, qb)

# -----------------------------------------------------------------------------
# 3D
# -----------------------------------------------------------------------------
def complex_quad_3d(func, np.ndarray a, np.ndarray b, np.ndarray c):
    samples = np.array([[0.333333333333333, 0.333333333333333, 0.225000000000000],
                        [0.797426985353087, 0.101286507323456, 0.125939180544827],
                        [0.101286507323456, 0.797426985353087, 0.125939180544827],
                        [0.101286507323456, 0.101286507323456, 0.125939180544827],
                        [0.470142064105115, 0.470142064105115, 0.132394152788506],
                        [0.470142064105115, 0.059715871789770, 0.132394152788506],
                        [0.059715871789770, 0.470142064105115, 0.132394152788506]],
                       dtype=np.float32)
    ab = b - a
    ac = c - a
    sum_ = 0.0
    for n in range(samples.shape[0]):
        x = a + samples[n, 0] * ab + samples[n, 1] * ac
        sum_ += samples[n, 2] * func(x)
    return sum_ * 0.5 * norm(np.cross(ab, ac))


def l_3d(float k, np.ndarray p, np.ndarray qa, np.ndarray qb, np.ndarray qc, bool p_on_element):
    if p_on_element:
        if k == 0.0:
            ab = qb - qa
            ac = qc - qa
            bc = qc - qb
            aopp = np.array([norm(ab), norm(bc), norm(ac)], dtype=np.float32)
            ap = p - qa
            bp = p - qb
            cp = p - qc
            ar0 = np.array([norm(ap), norm(bp), norm(cp)], dtype=np.float32)
            ara = np.array([norm(bp), norm(cp), norm(ap)], dtype=np.float32)
            result = 0.0
            for i in range(3):
                r0 = ar0[i]
                ra = ara[i]
                opp = aopp[i]
                if r0 < ra:
                    ra, r0 = r0, ra
                sr0 = r0**2
                sra = ra**2
                sopp = opp**2
                A = np.arccos((sra + sr0 - sopp) / (2.0 * ra * r0))
                B = np.arctan(ra*np.sin(A) / (r0 - ra*np.cos(A)))
                result += (r0*np.sin(B)*(np.log(np.tan(0.5*(B+A)))
                                         - np.log(np.tan(0.5*B))))
            return result / (4.0 * np.pi)
        else:
            def func(x):
                r = p - x
                R = norm(r)
                ikr = 1j * k * R
                return (np.exp(ikr) - 1.0) / R
            L0 = l_3d(0.0, p, qa, qb, qc, True)
            Lk = complex_quad_3d(func, qa, qb, p) + complex_quad_3d(func, qb, qc, p) \
                 + complex_quad_3d(func, qc, qa, p)
            return L0 + Lk / (4.0 * np.pi)
    else:
        if k == 0.0:
            def func(x):
                r = p - x
                R = norm(r)
                return 1.0 / R
            return complex_quad_3d(func, qa, qb, qc) / (4.0 * np.pi)
        else:
            def func(x):
                r = p - x
                R = norm(r)
                ikr = 1j * k * R
                return np.exp(ikr) / R
            return complex_quad_3d(func, qa, qb, qc) / (4.0 * np.pi)


def m_3d(float k, np.ndarray p, np.ndarray qa, np.ndarray qb, np.ndarray qc, bool p_on_element):
    if p_on_element:
        return 0.0
    else:
        if k == 0.0:
            def func(x):
                r = p - x
                R = norm(r)
                rnq = -np.dot(r, normal_3d(qa, qb, qc)) / R
                return -1.0 / np.dot(r, r) * rnq
            return complex_quad_3d(func, qa, qb, qc) / (4.0 * np.pi)
        else:
            def func(x):
                r = p - x
                R = norm(r)
                rnq = -np.dot(r, normal_3d(qa, qb, qc)) / R
                kr = k * R
                ikr = 1j * kr
                return rnq * (ikr - 1.0) * np.exp(ikr) / np.dot(r, r)
            return complex_quad_3d(func, qa, qb, qc) / (4.0 * np.pi)


def mt_3d(float k, np.ndarray p, np.ndarray vecp, np.ndarray qa, np.ndarray qb, np.ndarray qc, bool p_on_element):
    if p_on_element:
        return 0.0
    else:
        if k == 0.0:
            def func(x):
                r = p - x
                R = norm(r)
                rnp = np.dot(r, vecp) / R
                return -1.0 / np.dot(r, r) * rnp
            return complex_quad_3d(func, qa, qb, qc) / (4.0 * np.pi)
        else:
            def func(x):
                r = p - x
                R = norm(r)
                rnp = np.dot(r, vecp) / R
                ikr = 1j * k * R
                return rnp * (ikr - 1.0) * np.exp(ikr) / np.dot(r, r)
            return complex_quad_3d(func, qa, qb, qc) / (4.0 * np.pi)


def n_3d(float k, np.ndarray p, np.ndarray vecp, np.ndarray qa, np.ndarray qb, np.ndarray qc, bool p_on_element):
    if p_on_element:
        if k == 0.0:
            ab = qb - qa
            ac = qc - qa
            bc = qc - qb
            aopp = np.array([norm(ab), norm(bc), norm(ac)], dtype=np.float32)
            ap = p - qa
            bp = p - qb
            cp = p - qc
            ar0 = np.array([norm(ap), norm(bp), norm(cp)], dtype=np.float32)
            ara = np.array([norm(bp), norm(cp), norm(ap)], dtype=np.float32)
            result = 0.0
            for i in range(3):
                r0 = ar0[i]
                ra = ara[i]
                opp = aopp[i]
                if r0 < ra:
                    ra, r0 = r0, ra
                sr0 = r0**2
                sra = ra**2
                sopp = opp**2
                A = np.arccos((sra + sr0 - sopp) / (2.0 * ra * r0))
                B = np.arctan(ra*np.sin(A) / (r0 - ra*np.cos(A)))
                result += (np.cos(B+A) - np.cos(B)) / (r0 * np.sin(B))
            return result / (4.0 * np.pi)
        else:
            def func(x):
                r = p - x
                R = norm(r)
                vecq = normal_3d(qa, qb, qc)

                rnq = -np.dot(r, vecq) / R
                rnp = np.dot(r, vecp) / R
                dnpnq = np.dot(vecp, vecq)
                rnprnq = rnp * rnq
                rnpnq = -(dnpnq + rnprnq) / R

                kr = k * R
                ikr = 1j * kr
                fpg = 1.0 / R
                fpgr = ((ikr - 1.0) * np.exp(ikr) + 1.0) / np.dot(r, r)
                fpgrr = (np.exp(ikr) * (2.0 - 2.0*ikr - kr*kr) - 2.0) / (R * np.dot(r, r))

                return fpgr * rnpnq + fpgrr * rnprnq + (0.5*k*k) * fpg
            N0 = n_3d(0.0, p, vecp, qa, qb, qc, True)
            L0 = l_3d(0.0, p, qa, qb, qc, True)
            Nk = complex_quad_3d(func, qa, qb, p) + complex_quad_3d(func, qb, qc, p) + complex_quad_3d(func, qc, qa, p)
            return N0 - (0.5*k*k) * L0 + Nk / (4.0 * np.pi)
    else:
        if k == 0.0:
            return 0.0
        else:
            def func(x):
                r = p - x
                R = norm(r)
                vecq = normal_3d(qa, qb, qc)

                rnq = -np.dot(r, vecq) / R
                rnp = np.dot(r, vecp) / R
                dnpnq = np.dot(vecp, vecq)
                rnprnq = rnp * rnq
                rnpnq = -(dnpnq + rnprnq) / R

                kr = k * R
                ikr = 1j * kr
                fpgr = (ikr - 1.0) * np.exp(ikr) / np.dot(r, r)
                fpgrr = np.exp(ikr) * (2.0 - 2.0*ikr - kr*kr) / (R * np.dot(r, r))

                return fpgr * rnpnq + fpgrr * rnprnq

            return complex_quad_3d(func, qa, qb, qc) / (4.0 * np.pi)


# -----------------------------------------------------------------------------
# RAD
# -----------------------------------------------------------------------------
class CircularIntegratorPi(object):
    """
    Integrator class for integrating the upper half-circle or in other
    words integrate a function along the unit acr over angles
    theta in [0, pi].
    """
    samples = np.array([[0.980144928249, 5.061426814519E-02],
                        [0.898333238707, 0.111190517227],
                        [0.762766204958, 0.156853322939],
                        [0.591717321248, 0.181341891689],
                        [0.408282678752, 0.181341891689],
                        [0.237233795042, 0.156853322939],
                        [0.101666761293, 0.111190517227],
                        [1.985507175123E-02, 5.061426814519E-02]], dtype=np.float32)

    def __init__(self, segments):
        self.segments = segments
        nSamples = segments * self.samples.shape[0]
        self.rotationFactors = np.empty((nSamples, 2), dtype=np.float32)

        factor = np.pi / self.segments
        for i in range(nSamples):
            arcAbscissa = i // self.samples.shape[0] + self.samples[i % self.samples.shape[0], 0]
            arcAbscissa *= factor
            self.rotationFactors[i, :] = np.cos(arcAbscissa), np.sin(arcAbscissa)

    def integrate(self, func):
        sum = 0.0
        for n in range(self.rotationFactors.shape[0]):
            sum += self.samples[n % self.samples.shape[0], 1] * func(self.rotationFactors[n, :])
        return sum * np.pi / self.segments


def complex_quad_generator(func, np.ndarray start, np.ndarray end):
    """
    This is a variation on the basic complex quadrature function from the
    base class. The difference is, that the abscissa values y**2 have been
    substituted for x. Kirkup doesn't explain the details of why this
    is helpful for the case of this kind of 2D integral evaluation, but points
    to his PhD thesis and another reference that I have no access to.
    """
    samples = np.array([[0.980144928249, 5.061426814519E-02],
                        [0.898333238707, 0.111190517227],
                        [0.762766204958, 0.156853322939],
                        [0.591717321248, 0.181341891689],
                        [0.408282678752, 0.181341891689],
                        [0.237233795042, 0.156853322939],
                        [0.101666761293, 0.111190517227],
                        [1.985507175123E-02, 5.061426814519E-02]], dtype=np.float32)
    vec = end - start
    sum = 0.0
    for n in range(samples.shape[0]):
        x = start + samples[n, 0]**2 * vec
        sum += samples[n, 1] * func(x) * samples[n, 0]

    return 2.0 * sum * norm(vec)


def complex_quad_cone(func, np.ndarray start, np.ndarray end, int segments = 1):
    delta = 1.0 / segments * (end - start)
    sum = 0.0
    for s in range(segments):
        sum += complex_quad_2d(func, start + s * delta, start + (s + 1) * delta)

    return sum


def l_rad(float k, np.ndarray p, np.ndarray qa, np.ndarray qb, bool p_on_element):
    ab = qb - qa
    # subdivide circular integral into sections of
    # similar size as ab
    q = 0.5 * (qa + qb)
    nSections = 1 + int(q[0] * np.pi / norm(ab))
    if p_on_element:
        if k == 0.0:
            def generatorFunc(x):
                circle = CircularIntegratorPi(2 * nSections)
                r = x[0]
                z = x[1]
                p3 = np.array([p[0], 0.0, p[1]], dtype=np.float32)

                def circleFunc(x):
                    q3 = np.array([r * x[0], r * x[1], z], dtype=np.float32)
                    rr = q3 - p3
                    return 1.0 / norm(rr)

                return circle.integrate(circleFunc) * r / (2.0 * np.pi)

            return complex_quad_generator(generatorFunc, p, qa) + complex_quad_generator(generatorFunc, p, qb)

        else:
            def generatorFunc(x):
                circle = CircularIntegratorPi(2 * nSections)
                r = x[0]
                z = x[1]
                p3 = np.array([p[0], 0.0, p[1]], dtype=np.float32)

                def circleFunc(x):
                    q3 = np.array([r * x[0], r * x[1], z], dtype=np.float32)
                    rr = q3 - p3
                    RR = norm(rr)
                    return (np.exp(1.0j * k * RR) - 1.0) / RR

                return circle.integrate(circleFunc) * r / (2.0 * np.pi)

            return l_rad(0.0, p, qa, qb, True) + complex_quad_2d(generatorFunc, qa, qb)

    else:
        if k == 0.0:
            def generatorFunc(x):
                circle = CircularIntegratorPi(nSections)
                r = x[0]
                z = x[1]
                p3 = np.array([p[0], 0.0, p[1]], dtype=np.float32)

                def circleFunc(x):
                    q3 = np.array([r * x[0], r * x[1], z], dtype=np.float32)
                    rr = q3 - p3
                    return 1.0 / norm(rr)

                return circle.integrate(circleFunc) * r / (2.0 * np.pi)

            return complex_quad_2d(generatorFunc, qa, qb)

        else:
            def generatorFunc(x):
                circle = CircularIntegratorPi(nSections)
                r = x[0]
                z = x[1]
                p3 = np.array([p[0], 0.0, p[1]], dtype=np.float32)

                def circleFunc(x):
                    q3 = np.array([r * x[0], r * x[1], z], dtype=np.float32)
                    rr = q3 - p3
                    RR = norm(rr)
                    return np.exp(1.0j * k * RR) / RR

                return circle.integrate(circleFunc) * r / (2.0 * np.pi)

            return complex_quad_2d(generatorFunc, qa, qb)


def m_rad(float k, np.ndarray p, np.ndarray qa, np.ndarray qb, bool p_on_element):
    qab = qb - qa
    vec_q = normal_2d(qa, qb)

    # subdived circular integral into sections of
    # similar size as qab
    q = 0.5 * (qa + qb)
    nSections = 1 + int(q[0] * np.pi / norm(qab))

    if k == 0.0:
        def generatorFunc(x):
            circle = CircularIntegratorPi(nSections)
            r = x[0]
            z = x[1]
            p3 = np.array([p[0], 0.0, p[1]], dtype=np.float32)

            def circleFunc(x):
                q3 = np.array([r * x[0], r * x[1], z], dtype=np.float32)
                vec_q3 = np.array([vec_q[0] * x[0], vec_q[0] * x[1], vec_q[1]], dtype=np.float32)
                rr = q3 - p3

                return -np.dot(rr, vec_q3) / (norm(rr) * np.dot(rr, rr))

            return circle.integrate(circleFunc) * r / (2.0 * np.pi)

        if p_on_element:
            return complex_quad_2d(generatorFunc, qa, p) + complex_quad_2d(generatorFunc, p, qb)
        else:
            return complex_quad_2d(generatorFunc, qa, qb)

    else:
        def generatorFunc(x):
            circle = CircularIntegratorPi(nSections)
            r = x[0]
            z = x[1]
            p3 = np.array([p[0], 0.0, p[1]], dtype=np.float32)

            def circleFunc(x):
                q3 = np.array([r * x[0], r * x[1], z], dtype=np.float32)
                vec_q3 = np.array([vec_q[0] * x[0], vec_q[0] * x[1], vec_q[1]], dtype=np.float32)
                rr = q3 - p3
                RR = norm(rr)
                return (1j * k * RR - 1.0) * np.exp(1j * k * RR) * np.dot(rr, vec_q3) / (RR *  np.dot(rr, rr))

            return circle.integrate(circleFunc) * r / (2.0 * np.pi)

        if p_on_element:
            return complex_quad_2d(generatorFunc, qa, p) + complex_quad_2d(generatorFunc, p, qb)
        else:
            return complex_quad_2d(generatorFunc, qa, qb)


def mt_rad(float k, np.ndarray p, np.ndarray vecp, np.ndarray qa, np.ndarray qb, bool p_on_element):
    qab = qb - qa

    # subdived circular integral into sections of
    # similar size as qab
    q = 0.5 * (qa + qb)
    nSections = 1 + int(q[0] * np.pi / norm(qab))

    if k == 0.0:
        def generatorFunc(x):
            circle = CircularIntegratorPi(nSections)
            r = x[0]
            z = x[1]
            p3 = np.array([p[0], 0.0, p[1]], dtype=np.float32)

            def circleFunc(x):
                q3 = np.array([r * x[0], r * x[1], z], dtype=np.float32)
                rr = q3 - p3
                dotRnP = vecp[0] * rr[0] + vecp[1] * rr[2]
                return dotRnP / (norm(rr) * np.dot(rr, rr))

            return circle.integrate(circleFunc) * r / (2.0 * np.pi)

        if p_on_element:
            return complex_quad_2d(generatorFunc, qa, p) + complex_quad_2d(generatorFunc, p, qb)
        else:
            return complex_quad_2d(generatorFunc, qa, qb)

    else:
        def generatorFunc(x):
            circle = CircularIntegratorPi(nSections)
            r = x[0]
            z = x[1]
            p3 = np.array([p[0], 0.0, p[1]], dtype=np.float32)

            def circleFunc(x):
                q3 = np.array([r * x[0], r * x[1], z], dtype=np.float32)
                rr = q3 - p3
                RR = norm(rr)
                dotRnP = vecp[0] * rr[0] + vecp[1] * rr[2]
                return -(1j * k * RR - 1.0) * np.exp(1j * k * RR) * dotRnP / (RR *  np.dot(rr, rr))

            return circle.integrate(circleFunc) * r / (2.0 * np.pi)

        if p_on_element:
            return complex_quad_2d(generatorFunc, qa, p) + complex_quad_2d(generatorFunc, p, qb)
        else:
            return complex_quad_2d(generatorFunc, qa, qb)


def n_rad(float k, np.ndarray p, np.ndarray vecp, np.ndarray qa, np.ndarray qb, bool p_on_element):
    qab = qb - qa
    vec_q = normal_2d(qa, qb)

    # subdived circular integral into sections of
    # similar size as qab
    q = 0.5 * (qa + qb)
    nSections = 1 + int(q[0] * np.pi / norm(qab))

    if p_on_element:
        if k == 0.0:
            vecp3 = np.array([vecp[0], 0.0, vecp[1]], dtype=np.float32)
            def coneFunc(x, direction):
                circle = CircularIntegratorPi(nSections)
                r = x[0]
                z = x[1]
                p3 = np.array([p[0], 0.0, p[1]], dtype=np.float32)

                def circleFunc(x):
                    q3 = np.array([r * x[0], r * x[1], z], dtype=np.float32)
                    vec_q3 = np.sqrt(0.5) * np.array([x[0], x[1], direction], dtype=np.float32)
                    dnpnq = np.dot(vecp3, vec_q3)
                    rr = q3 - p3
                    RR = norm(rr)
                    dotRNP = np.dot(rr, vecp3)
                    dotRNQ = -np.dot(rr, vec_q3)
                    RNPRNQ = dotRNP * dotRNQ / np.dot(rr, rr)
                    return (dnpnq + 3.0 * RNPRNQ) / (RR * np.dot(rr, rr))

                return circle.integrate(circleFunc) * r / (2.0 * np.pi)

            lenAB = norm(qab)
            # deal with the cone at the qa side of the generator
            direction = np.sign(qa[1] - qb[1])
            if direction == 0.0:
                direction = 1.0
            tip_a = np.array([0.0, qa[1] + direction * qa[0]], dtype=np.float32)
            nConeSectionsA = int(qa[0] * np.sqrt(2.0) / lenAB) + 1
            coneValA = complex_quad_cone(lambda x: coneFunc(x, direction), qa, tip_a, nConeSectionsA)

            # deal with the cone at the qb side of the generator
            direction = np.sign(qb[1] - qa[1])
            if direction == 0.0:
                direction = -1.0
            tip_b = np.array([0.0, qb[1] + direction * qb[0]], dtype=np.float32)
            nConeSectionsB = int(qb[0] * np.sqrt(2.0) / lenAB) + 1
            coneValB = complex_quad_cone(lambda x: coneFunc(x, direction), qb, tip_b, nConeSectionsB)

            return -(coneValA + coneValB)

        else:
            def generatorFunc(x):
                circle = CircularIntegratorPi(nSections)
                r = x[0]
                z = x[1]
                p3 = np.array([p[0], 0.0, p[1]], dtype=np.float32)

                def circleFunc(x):
                    q3 = np.array([r * x[0], r * x[1], z], dtype=np.float32)
                    vec_q3 = np.array([vec_q[0] * x[0], vec_q[0] * x[1], vec_q[1]], dtype=np.float32)
                    rr = q3 - p3
                    RR = norm(rr)
                    DNPNQ = vecp[0] * vec_q3[0] + vecp[1] * vec_q3[2]
                    dotRnP = vecp[0] * rr[0] + vecp[1] * rr[2]
                    dotRnQ = -np.dot(rr, vec_q3)
                    RNPRNQ = dotRnP * dotRnQ / np.dot(rr, rr)
                    RNPNQ = -(DNPNQ + RNPRNQ) / RR
                    IKR = 1j * k * RR
                    FPG0 = 1.0 / RR
                    FPGR = np.exp(IKR) / np.dot(rr, rr) * (IKR - 1.0)
                    FPGR0 = -1.0 / np.dot(rr, rr)
                    FPGRR = np.exp(IKR) * (2.0 - 2.0 * IKR - (k*RR)**2) / (RR * np.dot(rr, rr))
                    FPGRR0 = 2.0 / (RR * np.dot(rr, rr))
                    return (FPGR - FPGR0) * RNPNQ + (FPGRR - FPGRR0) * RNPRNQ \
                        + k**2 * FPG0 / 2.0

                return circle.integrate(circleFunc) * r / (2.0 * np.pi)

            return n_rad(0.0, p, vecp, qa, qb, True) \
                - k ** 2 * l_rad(0.0, p, qa, qb, True) / 2.0 \
                + complex_quad_2d(generatorFunc, qa, p) + complex_quad_2d(generatorFunc, p, qb)

    else:
        if k == 0.0:
            def generatorFunc(x):
                circle = CircularIntegratorPi(nSections)
                r = x[0]
                z = x[1]
                p3 = np.array([p[0], 0.0, p[1]], dtype=np.float32)

                def circleFunc(x):
                    q3 = np.array([r * x[0], r * x[1], z], dtype=np.float32)
                    vec_q3 = np.array([vec_q[0] * x[0], vec_q[0] * x[1], vec_q[1]], dtype=np.float32)
                    rr = q3 - p3
                    RR = norm(rr)
                    DNPNQ = vecp[0] * vec_q3[0] + vecp[1] * vec_q3[2]
                    dotRnP = vecp[0] * rr[0] + vecp[1] * rr[2]
                    dotRnQ = -np.dot(rr, vec_q3)
                    RNPRNQ = dotRnP * dotRnQ / np.dot(rr, rr)
                    RNPNQ = -(DNPNQ + RNPRNQ) / RR
                    IKR = 1j * k * RR
                    FPGR = -1.0 / np.dot(rr, rr)
                    FPGRR = 2.0 / (RR * np.dot(rr, rr))
                    return FPGR * RNPNQ + FPGRR * RNPRNQ

                return circle.integrate(circleFunc) * r / (2.0 * np.pi)

            return 0.0 # complex_quad_2d(generatorFunc, qa, qb)
        else:
            def generatorFunc(x):
                circle = CircularIntegratorPi(nSections)
                r = x[0]
                z = x[1]
                p3 = np.array([p[0], 0.0, p[1]], dtype=np.float32)

                def circleFunc(x):
                    q3 = np.array([r * x[0], r * x[1], z], dtype=np.float32)
                    vec_q3 = np.array([vec_q[0] * x[0], vec_q[0] * x[1], vec_q[1]], dtype=np.float32)
                    rr = q3 - p3
                    RR = norm(rr)
                    DNPNQ = vecp[0] * vec_q3[0] + vecp[1] * vec_q3[2]
                    dotRnP = vecp[0] * rr[0] + vecp[1] * rr[2]
                    dotRnQ = -np.dot(rr, vec_q3)
                    RNPRNQ = dotRnP * dotRnQ / np.dot(rr, rr)
                    RNPNQ = -(DNPNQ + RNPRNQ) / RR
                    IKR = 1j * k * RR
                    FPGR = np.exp(IKR) / np.dot(rr, rr) * (IKR - 1.0)
                    FPGRR = np.exp(IKR) * (2.0 - 2.0 * IKR - (k*RR)**2) / (RR * np.dot(rr, rr))
                    return FPGR * RNPNQ + FPGRR * RNPRNQ

                return circle.integrate(circleFunc) * r / (2.0 * np.pi)

            return complex_quad_2d(generatorFunc, qa, qb)
