import numpy as np
from numpy.linalg import norm
from .helmholtz_integrals_2d import complex_quad
from .helmholtz_integrals_rad_c import l_rad_c, m_rad_c, mt_rad_c, n_rad_c
from .normals import normal_2d


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
            arcAbscissa = i / self.samples.shape[0] + self.samples[i % self.samples.shape[0], 0]
            arcAbscissa *= factor
            self.rotationFactors[i, :] = np.cos(arcAbscissa), np.sin(arcAbscissa)

    def integrate(self, func):
        sum = 0.0
        for n in range(self.rotationFactors.shape[0]):
            sum += self.samples[n % self.samples.shape[0], 1] * func(self.rotationFactors[n, :])
        return sum * np.pi / self.segments


def complex_quad_generator(func, start, end):
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


def complex_quad_cone(func, start, end, segments = 1):
    delta = 1.0 / segments * (end - start)
    sum = 0.0
    for s in range(segments):
        sum += complex_quad(func, start + s * delta, start + (s + 1) * delta)

    return sum


bOptimized = True


def l_rad(k, p, qa, qb, p_on_element):
    if bOptimized:
        return l_rad_c(k, p, qa, qb, p_on_element)
    qab = qb - qa
    # subdived circular integral into sections of
    # similar size as qab
    q = 0.5 * (qa + qb)
    nSections = 1 + int(q[0] * np.pi / norm(qab))
    if p_on_element:
        ap = p - qa

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

            return l_rad(0.0, p, qa, qb, True) + complex_quad(generatorFunc, qa, qb)

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

            return complex_quad(generatorFunc, qa, qb)

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

            return complex_quad(generatorFunc, qa, qb)


def m_rad(k, p, qa, qb, p_on_element):
    if bOptimized:
        return m_rad_c(k, p, qa, qb, p_on_element)
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
            return complex_quad(generatorFunc, qa, p) + complex_quad(generatorFunc, p, qb)
        else:
            return complex_quad(generatorFunc, qa, qb)

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
            return complex_quad(generatorFunc, qa, p) + complex_quad(generatorFunc, p, qb)
        else:
            return complex_quad(generatorFunc, qa, qb)


def mt_rad(k, p, vecp, qa, qb, p_on_element):
    if bOptimized:
        return mt_rad_c(k, p, vecp, qa, qb, p_on_element)
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
                dotRnP = vecp[0] * rr[0] + vec[1] * rr[2]
                return dotRnP / (norm(rr) * np.dot(rr, rr))

            return circle.integrate(circleFunc) * r / (2.0 * np.pi)

        if p_on_element:
            return complex_quad(generatorFunc, qa, p) + complex_quad(generatorFunc, p, qb)
        else:
            return complex_quad(generatorFunc, qa, qb)

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
            return complex_quad(generatorFunc, qa, p) + complex_quad(generatorFunc, p, qb)
        else:
            return complex_quad(generatorFunc, qa, qb)


def n_rad(k, p, vecp, qa, qb, p_on_element):
    if bOptimized:
        return n_rad_c(k, p, vecp, qa, qb, p_on_element)
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
                + complex_quad(generatorFunc, qa, p) + complex_quad(generatorFunc, p, qb)

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

            return complex_quad(generatorFunc, qa, qb)
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

            return complex_quad(generatorFunc, qa, qb)
