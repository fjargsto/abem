from .native_interface import Float2, Complex, intops
from ctypes import c_float, c_bool, byref
import numpy as np


def l_rad_c(k, p, qa, qb, p_on_element):
    result = Complex()
    pp = Float2(p[0], p[1])
    a = Float2(qa[0], qa[1])
    b = Float2(qb[0], qb[1])
    x = c_bool(p_on_element)
    intops.ComputeL_RAD(c_float(k), pp, a, b, x, byref(result))

    return np.complex64(result.re+result.im*1j)


def m_rad_c(k, p, qa, qb, p_on_element):
    result = Complex()
    pp = Float2(p[0], p[1])
    a = Float2(qa[0], qa[1])
    b = Float2(qb[0], qb[1])
    x = c_bool(p_on_element)
    intops.ComputeM_RAD(c_float(k), pp, a, b, x, byref(result))

    return np.complex64(result.re+result.im*1j)


def mt_rad_c(k, p, vec_p, qa, qb, p_on_element):
    result = Complex()
    pp = Float2(p[0], p[1])
    vec_pp = Float2(vec_p[0], vec_p[1])
    a = Float2(qa[0], qa[1])
    b = Float2(qb[0], qb[1])
    x = c_bool(p_on_element)
    intops.ComputeMt_RAD(c_float(k), pp, vec_pp, a, b, x, byref(result))

    return np.complex64(result.re+result.im*1j)


def n_rad_c(k, p, vec_p, qa, qb, p_on_element):
    result = Complex()
    pp = Float2(p[0], p[1])
    vec_pp = Float2(vec_p[0], vec_p[1])
    a = Float2(qa[0], qa[1])
    b = Float2(qb[0], qb[1])
    x = c_bool(p_on_element)
    intops.ComputeN_RAD(c_float(k), pp, vec_pp, a, b, x, byref(result))

    return np.complex64(result.re+result.im*1j)
