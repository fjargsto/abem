import numpy as np
from numpy.linalg import norm
from scipy.special import hankel1
from .normals import normal_2d
from .helmholtz_integrals_2d_c import l_2d_c, m_2d_c, mt_2d_c, n_2d_c

bOptimized = True

def complex_quad(func, start, end):
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


def l_2d_p(k, p, qa, qb, p_on_element):
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
                complex_quad(func, qa, p)
                + complex_quad(func, p, qa)
                + l_2d(0.0, p, qa, qb, True)
            )
    else:
        if k == 0.0:
            result = -0.5 / np.pi * complex_quad(lambda q: np.log(norm(p - q)), qa, qb)
        else:
            result =  0.25j * complex_quad(lambda q: hankel1(0, k * norm(p - q)), qa, qb)

    return result


def l_2d(k, p, qa, qb, p_on_element):
    result_p = l_2d_p(k, p, qa, qb, p_on_element)
    result_c = l_2d_c(k, p, qa, qb, p_on_element)
    assert abs(result_c - result_p) < 0.0000001
    if bOptimized:
        return result_c
    else:
        return result_p


def m_2d_p(k, p, qa, qb, p_on_element):
    vecq = normal_2d(qa, qb)
    if p_on_element:
        result = 0.0
    else:
        if k == 0.0:

            def func(x):
                r = p - x
                return np.dot(r, vecq) / np.dot(r, r)

            result = -0.5 / np.pi * complex_quad(func, qa, qb)
        else:

            def func(x):
                r = p - x
                R = norm(r)
                return hankel1(1, k * R) * np.dot(r, vecq) / R

            result = 0.25j * k * complex_quad(func, qa, qb)

    return result


def m_2d(k, p, qa, qb, p_on_element):
    result_p = m_2d_p(k, p, qa, qb, p_on_element)
    result_c = m_2d_c(k, p, qa, qb, p_on_element)
    assert abs(result_c - result_p) < 0.0000001
    if bOptimized:
        return result_c
    else:
        return result_p


def mt_2d_p(k, p, vecp, qa, qb, p_on_element):
    if bOptimized:
        return mt_2d_c(k, p, vecp, qa, qb, p_on_element)
    if p_on_element:
        return 0.0
    else:
        if k == 0.0:

            def func(x):
                r = p - x
                return np.dot(r, vecp) / np.dot(r, r)

            return -0.5 / np.pi * complex_quad(func, qa, qb)
        else:

            def func(x):
                r = p - x
                R = norm(r)
                return hankel1(1, k * R) * np.dot(r, vecp) / R

            return -0.25j * k * complex_quad(func, qa, qb)


def mt_2d(k, p, vecp, qa, qb, p_on_element):
    if bOptimized:
        return mt_2d_c(k, p, vecp, qa, qb, p_on_element)
    else:
        return mt_2d_p(k, p, vecp, qa, qb, p_on_element)


def n_2d_p(k, p, vecp, qa, qb, p_on_element):
    if bOptimized:
        return n_2d_c(k, p, vecp, qa, qb, p_on_element)
    qab = qb - qa
    if p_on_element:
        ra = norm(p - qa)
        rb = norm(p - qb)
        re = norm(qab)
        if k == 0.0:
            return -(1.0 / ra + 1.0 / rb) / (re * 2.0 * np.pi) * re
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

            return (
                n_2d(0.0, p, vecp, qa, qb, True)
                - 0.5 * k_sqr * l_2d(0.0, p, qa, qb, True)
                + complex_quad(func, qa, p)
                + complex_quad(func, p, qb)
            )
    else:
        vecq = normal_2d(qa, qb)
        un = np.dot(vecp, vecq)
        if k == 0.0:

            def func(x):
                r = p - x
                R2 = np.dot(r, r)
                drdudrdn = -np.dot(r, vecq) * np.dot(r, vecp) / R2
                return (un + 2.0 * drdudrdn) / R2

            return 0.5 / np.pi * complex_quad(func, qa, qb)
        else:

            def func(x):
                r = p - x
                drdudrdn = -np.dot(r, vecq) * np.dot(r, vecp) / np.dot(r, r)
                R = norm(r)
                return (
                    hankel1(1, k * R) / R * (un + 2.0 * drdudrdn)
                    - k * hankel1(0, k * R) * drdudrdn
                )

            return 0.25j * k * complex_quad(func, qa, qb)


def n_2d(k, p, vecp, qa, qb, p_on_element):
    if bOptimized:
        return n_2d_c(k, p, vecp, qa, qb, p_on_element)
    else:
        return n_2d_p(k, p, vecp, qa, qb, p_on_element)
