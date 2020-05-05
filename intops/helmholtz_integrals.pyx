from intops cimport *
import numpy as np


cpdef hankel1_c(int order, float x):
    cdef Complex z
    Hankel1(order, x, &z)
    return np.complex64(z.re + z.im * 1j)

# -----------------------------------------------------------------------------
# 2D
# -----------------------------------------------------------------------------
cpdef l_2d_c(float k, p, qa, qb, bool p_on_element):
    cdef Complex result
    cdef Float2 cp
    cp.x = p[0]
    cp.y = p[1]
    cdef Float2 a
    a.x = qa[0]
    a.y = qa[1]
    cdef Float2 b
    b.x = qb[0]
    b.y = qb[1]
    ComputeL_2D(k, cp, a, b, p_on_element, &result)
    return np.complex64(result.re + result.im * 1j)


cpdef m_2d_c(float k, p, qa, qb, bool p_on_element):
    cdef Complex result
    cdef Float2 cp
    cp.x = p[0]
    cp.y = p[1]
    cdef Float2 a
    a.x = qa[0]
    a.y = qa[1]
    cdef Float2 b
    b.x = qb[0]
    b.y = qb[1]
    ComputeM_2D(k, cp, a, b, p_on_element, &result)
    return np.complex64(result.re + result.im * 1j)


cpdef mt_2d_c(float k, p, normal_p, qa, qb, bool p_on_element):
    cdef Complex result
    cdef Float2 cp
    cp.x = p[0]
    cp.y = p[1]
    cdef Float2 c_normal_p
    c_normal_p.x = normal_p[0]
    c_normal_p.y = normal_p[1]
    cdef Float2 a
    a.x = qa[0]
    a.y = qa[1]
    cdef Float2 b
    b.x = qb[0]
    b.y = qb[1]
    ComputeMt_2D(k, cp, c_normal_p, a, b, p_on_element, &result)
    return np.complex64(result.re + result.im * 1j)


cpdef n_2d_c(float k, p, normal_p, qa, qb, bool p_on_element):
    cdef Complex result
    cdef Float2 cp
    cp.x = p[0]
    cp.y = p[1]
    cdef Float2 c_normal_p
    c_normal_p.x = normal_p[0]
    c_normal_p.y = normal_p[1]
    cdef Float2 a
    a.x = qa[0]
    a.y = qa[1]
    cdef Float2 b
    b.x = qb[0]
    b.y = qb[1]
    ComputeN_2D(k, cp, c_normal_p, a, b, p_on_element, &result)
    return np.complex64(result.re + result.im * 1j)

# -----------------------------------------------------------------------------
# 3D
# -----------------------------------------------------------------------------
cpdef l_3d_c(float k, p, qa, qb, qc, bool p_on_element):

    cdef Complex result
    cdef Float3 cp
    cp.x = p[0]
    cp.y = p[1]
    cp.z = p[2]
    cdef Float3 a
    a.x = qa[0]
    a.y = qa[1]
    a.z = qa[2]
    cdef Float3 b
    b.x = qb[0]
    b.y = qb[1]
    b.z = qb[2]
    cdef Float3 c
    c.x = qc[0]
    c.y = qc[1]
    c.z = qc[2]
    ComputeL_3D(k, cp, a, b, c, p_on_element, &result)
    return np.complex64(result.re+result.im*1j)


cpdef m_3d_c(float k, p, qa, qb, qc, bool p_on_element):
    cdef Complex result
    cdef Float3 cp
    cp.x = p[0]
    cp.y = p[1]
    cp.z = p[2]
    cdef Float3 a
    a.x = qa[0]
    a.y = qa[1]
    a.z = qa[2]
    cdef Float3 b
    b.x = qb[0]
    b.y = qb[1]
    b.z = qb[2]
    cdef Float3 c
    c.x = qc[0]
    c.y = qc[1]
    c.z = qc[2]
    ComputeM_3D(k, cp, a, b, c, p_on_element, &result)
    return np.complex64(result.re+result.im*1j)


cpdef mt_3d_c(float k, p, vec_p, qa, qb, qc, bool p_on_element):
    cdef Complex result
    cdef Float3 cp
    cp.x = p[0]
    cp.y = p[1]
    cp.z = p[2]
    cdef Float3 a
    a.x = qa[0]
    a.y = qa[1]
    a.z = qa[2]
    cdef Float3 b
    b.x = qb[0]
    b.y = qb[1]
    b.z = qb[2]
    cdef Float3 c
    c.x = qc[0]
    c.y = qc[1]
    c.z = qc[2]
    cdef Float3 vp
    vp.x = vec_p[0]
    vp.y = vec_p[1]
    vp.z = vec_p[2]
    ComputeMt_3D(k, cp, vp, a, b, c, p_on_element, &result)
    return np.complex64(result.re+result.im*1j)


cpdef n_3d_c(float k, p, vec_p, qa, qb, qc, bool p_on_element):
    cdef Complex result
    cdef Float3 cp
    cp.x = p[0]
    cp.y = p[1]
    cp.z = p[2]
    cdef Float3 a
    a.x = qa[0]
    a.y = qa[1]
    a.z = qa[2]
    cdef Float3 b
    b.x = qb[0]
    b.y = qb[1]
    b.z = qb[2]
    cdef Float3 c
    c.x = qc[0]
    c.y = qc[1]
    c.z = qc[2]
    cdef Float3 vp
    vp.x = vec_p[0]
    vp.y = vec_p[1]
    vp.z = vec_p[2]
    ComputeN_3D(k, cp, vp, a, b, c, p_on_element, &result)
    return np.complex64(result.re+result.im*1j)


# -----------------------------------------------------------------------------
# RAD
# -----------------------------------------------------------------------------
def l_rad_c(k, p, qa, qb, p_on_element):
    cdef Complex result
    cdef Float2 cp
    cp.x = p[0]
    cp.y = p[1]
    cdef Float2 a
    a.x = qa[0]
    a.y = qa[1]
    cdef Float2 b
    b.x = qb[0]
    b.y = qb[1]
    ComputeL_RAD(k, cp, a, b, p_on_element, &result)
    return np.complex64(result.re+result.im*1j)


cpdef m_rad_c(float k, p, qa, qb, bool p_on_element):
    cdef Complex result
    cdef Float2 cp
    cp.x = p[0]
    cp.y = p[1]
    cdef Float2 a
    a.x = qa[0]
    a.y = qa[1]
    cdef Float2 b
    b.x = qb[0]
    b.y = qb[1]
    ComputeM_RAD(k, cp, a, b, p_on_element, &result)
    return np.complex64(result.re+result.im*1j)


cpdef mt_rad_c(float k, p, vec_p, qa, qb, bool p_on_element):
    cdef Complex result
    cdef Float2 cp
    cp.x = p[0]
    cp.y = p[1]
    cdef Float2 c_normal_p
    c_normal_p.x = vec_p[0]
    c_normal_p.y = vec_p[1]
    cdef Float2 a
    a.x = qa[0]
    a.y = qa[1]
    cdef Float2 b
    b.x = qb[0]
    b.y = qb[1]
    ComputeMt_RAD(k, cp, c_normal_p, a, b, p_on_element, &result)
    return np.complex64(result.re+result.im*1j)


cpdef n_rad_c(float k, p, vec_p, qa, qb, bool p_on_element):
    cdef Complex result
    cdef Float2 cp
    cp.x = p[0]
    cp.y = p[1]
    cdef Float2 c_normal_p
    c_normal_p.x = vec_p[0]
    c_normal_p.y = vec_p[1]
    cdef Float2 a
    a.x = qa[0]
    a.y = qa[1]
    cdef Float2 b
    b.x = qb[0]
    b.y = qb[1]
    ComputeN_RAD(k, cp, c_normal_p, a, b, p_on_element, &result)
    return np.complex64(result.re+result.im*1j)
