# cython: profile=True
# cython: language_level=3, boundscheck=False
from iops_cpp cimport *
import numpy as np
cimport numpy as np

np.import_array()

cpdef hankel1_c(int order, float x):
    cdef Complex z
    Hankel1(order, x, &z)
    return np.complex64(z.re + z.im * 1j)

# -----------------------------------------------------------------------------
# 2D
# -----------------------------------------------------------------------------
cpdef l_2d(float k, float[:] p, float[:] qa, float[:] qb, bool p_on_element):
    cdef Complex result
    cdef Float2 *cp = <Float2*>&p[0]
    cdef Float2 *a = <Float2*>&qa[0]
    cdef Float2 *b = <Float2*>&qb[0]
    L_2D(k, cp, a, b, p_on_element, &result)
    return np.complex64(result.re + result.im * 1j)


cpdef l_2d_on_k0(float[:] qa, float[:] qb):
    cdef Complex result
    cdef Float2 *a = <Float2*>&qa[0]
    cdef Float2 *b = <Float2*>&qb[0]
    L_2D_ON_K0(a, b, &result)
    return np.complex64(result.re + result.im * 1j)


cpdef l_2d_on(float k, float[:] p, float[:] qa, float[:] qb):
    cdef Complex result
    cdef Float2 *cp = <Float2*>&p[0]
    cdef Float2 *a = <Float2*>&qa[0]
    cdef Float2 *b = <Float2*>&qb[0]
    L_2D_ON(k, cp, a, b, &result)
    return np.complex64(result.re + result.im * 1j)


cpdef l_2d_off_k0(float[:] p, float[:] qa, float[:] qb):
    cdef Complex result
    cdef Float2 *cp = <Float2*>&p[0]
    cdef Float2 *a = <Float2*>&qa[0]
    cdef Float2 *b = <Float2*>&qb[0]
    L_2D_OFF_K0(cp, a, b, &result)
    return np.complex64(result.re + result.im * 1j)


cpdef l_2d_off(float k, float[:] p, float[:] qa, float[:] qb):
    cdef Complex result
    cdef Float2 *cp = <Float2*>&p[0]
    cdef Float2 *a = <Float2*>&qa[0]
    cdef Float2 *b = <Float2*>&qb[0]
    L_2D_OFF(k, cp, a, b, &result)
    return np.complex64(result.re + result.im * 1j)


cpdef m_2d(float k, float[:] p, float[:] qa, float[:] qb, bool p_on_element):
    cdef Complex result
    cdef Float2 *cp = <Float2*>&p[0]
    cdef Float2 *a = <Float2*>&qa[0]
    cdef Float2 *b = <Float2*>&qb[0]
    M_2D(k, cp, a, b, p_on_element, &result)
    return np.complex64(result.re + result.im * 1j)


cpdef m_2d_off_k0(float[:] p, float[:] qa, float[:] qb):
    cdef Complex result
    cdef Float2 *cp = <Float2*>&p[0]
    cdef Float2 *a = <Float2*>&qa[0]
    cdef Float2 *b = <Float2*>&qb[0]
    M_2D_OFF_K0(cp, a, b, &result)
    return np.complex64(result.re + result.im * 1j)


cpdef m_2d_off(float k, float[:] p, float[:] qa, float[:] qb):
    cdef Complex result
    cdef Float2 *cp = <Float2*>&p[0]
    cdef Float2 *a = <Float2*>&qa[0]
    cdef Float2 *b = <Float2*>&qb[0]
    M_2D_OFF(k, cp, a, b, &result)
    return np.complex64(result.re + result.im * 1j)


cpdef mt_2d(float k, float[:] p, float[:] normal_p, float[:] qa, float[:] qb, bool p_on_element):
    cdef Complex result
    cdef Float2 *cp = <Float2*>&p[0]
    cdef Float2 *c_normal_p = <Float2*>&normal_p[0]
    cdef Float2 *a = <Float2*>&qa[0]
    cdef Float2 *b = <Float2*>&qb[0]
    MT_2D(k, cp, c_normal_p, a, b, p_on_element, &result)
    return np.complex64(result.re + result.im * 1j)


cpdef mt_2d_off_k0(float[:] p, float[:] normal_p, float[:] qa, float[:] qb):
    cdef Complex result
    cdef Float2 *cp = <Float2*>&p[0]
    cdef Float2 *c_normal_p = <Float2*>&normal_p[0]
    cdef Float2 *a = <Float2*>&qa[0]
    cdef Float2 *b = <Float2*>&qb[0]
    MT_2D_OFF_K0(cp, c_normal_p, a, b, &result)
    return np.complex64(result.re + result.im * 1j)


cpdef mt_2d_off(float k, float[:] p, float[:] normal_p, float[:] qa, float[:] qb):
    cdef Complex result
    cdef Float2 *cp = <Float2*>&p[0]
    cdef Float2 *c_normal_p = <Float2*>&normal_p[0]
    cdef Float2 *a = <Float2*>&qa[0]
    cdef Float2 *b = <Float2*>&qb[0]
    MT_2D_OFF(k, cp, c_normal_p, a, b, &result)
    return np.complex64(result.re + result.im * 1j)


cpdef n_2d(float k, float[:] p, float[:] normal_p, float[:] qa, float[:] qb, bool p_on_element):
    cdef Complex result
    cdef Float2 *cp = <Float2*>&p[0]
    cdef Float2 *c_normal_p = <Float2*>&normal_p[0]
    cdef Float2 *a = <Float2*>&qa[0]
    cdef Float2 *b = <Float2*>&qb[0]
    N_2D(k, cp, c_normal_p, a, b, p_on_element, &result)
    return np.complex64(result.re + result.im * 1j)


cpdef n_2d_on_k0(float[:] qa, float[:] qb):
    cdef Complex result
    cdef Float2 *a = <Float2*>&qa[0]
    cdef Float2 *b = <Float2*>&qb[0]
    N_2D_ON_K0(a, b, &result)
    return np.complex64(result.re + result.im * 1j)


cpdef n_2d_on(float k, float[:] p, float[:] normal_p, float[:] qa, float[:] qb):
    cdef Complex result
    cdef Float2 *cp = <Float2*>&p[0]
    cdef Float2 *c_normal_p = <Float2*>&normal_p[0]
    cdef Float2 *a = <Float2*>&qa[0]
    cdef Float2 *b = <Float2*>&qb[0]
    N_2D_ON(k, cp, c_normal_p, a, b, &result)
    return np.complex64(result.re + result.im * 1j)


cpdef n_2d_off_k0(float[:] p, float[:] normal_p, float[:] qa, float[:] qb):
    cdef Complex result
    cdef Float2 *cp = <Float2*>&p[0]
    cdef Float2 *c_normal_p = <Float2*>&normal_p[0]
    cdef Float2 *a = <Float2*>&qa[0]
    cdef Float2 *b = <Float2*>&qb[0]
    N_2D_OFF_K0(cp, c_normal_p, a, b, &result)
    return np.complex64(result.re + result.im * 1j)


cpdef n_2d_off(float k, float[:] p, float[:] normal_p, float[:] qa, float[:] qb):
    cdef Complex result
    cdef Float2 *cp = <Float2*>&p[0]
    cdef Float2 *c_normal_p = <Float2*>&normal_p[0]
    cdef Float2 *a = <Float2*>&qa[0]
    cdef Float2 *b = <Float2*>&qb[0]
    N_2D_OFF(k, cp, c_normal_p, a, b, &result)
    return np.complex64(result.re + result.im * 1j)


cpdef boundary_matrices_2d(float k, mu_, float[:, :, :] edges_, np.complex64_t[:, :] A, np.complex64_t[:, :] B,
                           orientation_):
    cdef Complex mu = Complex(mu_.real, mu_.imag)
    cdef LineSegment *p_edges = <LineSegment*>&edges_[0, 0, 0]
    cdef float orientation
    if orientation_ == "interior":
        orientation = -1.0
    elif orientation_ == "exterior":
        orientation = 1.0
    else:
        assert False, "Invalid orientation: {}".format(orientation_)
    with nogil:
        BOUNDARY_MATRICES_2D(k, &mu, p_edges, <Complex*>&A[0, 0], <Complex*>&B[0, 0], edges_.shape[0], orientation)

cpdef compute_solution_matrices(float k, samples, edges):
    L = np.empty((samples.shape[0], edges.shape[0]), dtype=np.complex64)
    M = np.empty((samples.shape[0], edges.shape[0]), dtype=np.complex64)
    cdef np.complex64_t[:, :] viewL = L
    cdef np.complex64_t[:, :] viewM = M
    cdef float[:, :] viewSamples = samples
    cdef float[:, :, :] viewEdges = edges
    cdef float orientation
    cdef unsigned int num_samples = samples.shape[0]
    cdef unsigned int num_edges= edges.shape[0]
    with nogil:
        SOLUTION_MATRICES_2D(k, <Float2*>&viewSamples[0, 0], <LineSegment*>&viewEdges[0, 0, 0],
                             <Complex*>&viewL[0, 0], <Complex*>&viewM[0, 0], num_samples, num_edges)
    return L, M

cpdef compute_sample_phi(solution, samples, edges):
    phi = np.empty(samples.shape[0], dtype=np.complex64)
    cdef float k = solution.k
    cdef np.complex64_t[:] solution_phi = solution.phis
    cdef np.complex64_t[:] solution_v = solution.velocities
    cdef float[:, :] viewSamples = samples
    cdef float[:, :, :] viewEdges = edges
    cdef unsigned int num_samples = samples.shape[0]
    cdef unsigned int num_edges = edges.shape[0]
    cdef np.complex64_t[:] viewPhi = phi
    with nogil:
        SAMPLE_PHI_2D(k, <Float2*>&viewSamples[0, 0], <LineSegment*>&viewEdges[0, 0, 0], num_samples, num_edges,
                      <Complex*>&solution_phi[0], <Complex*>&solution_v[0],  <Complex*>&viewPhi[0])

    return phi


# -----------------------------------------------------------------------------
# 3D
# -----------------------------------------------------------------------------
cpdef l_3d(float k, float[:] p, float[:] qa, float[:] qb, float[:] qc, bool p_on_element):
    cdef Complex result
    cdef Float3 *cp = <Float3*>&p[0]
    cdef Float3 *a = <Float3*>&qa[0]
    cdef Float3 *b = <Float3*>&qb[0]
    cdef Float3 *c = <Float3*>&qc[0]
    ComputeL_3D(k, cp[0], a[0], b[0], c[0], p_on_element, &result)
    return np.complex64(result.re+result.im*1j)


cpdef m_3d(float k, float[:] p, float[:] qa, float[:] qb, float[:] qc, bool p_on_element):
    cdef Complex result
    cdef Float3 *cp = <Float3*>&p[0]
    cdef Float3 *a = <Float3*>&qa[0]
    cdef Float3 *b = <Float3*>&qb[0]
    cdef Float3 *c = <Float3*>&qc[0]
    ComputeM_3D(k, cp[0], a[0], b[0], c[0], p_on_element, &result)
    return np.complex64(result.re+result.im*1j)


cpdef mt_3d(float k, float[:] p, float[:] vec_p, float[:] qa, float[:] qb, float[:] qc, bool p_on_element):
    cdef Complex result
    cdef Float3 *cp = <Float3*>&p[0]
    cdef Float3 *a = <Float3*>&qa[0]
    cdef Float3 *b = <Float3*>&qb[0]
    cdef Float3 *c = <Float3*>&qc[0]
    cdef Float3 *vp = <Float3*>&vec_p[0]
    ComputeMt_3D(k, cp[0], vp[0], a[0], b[0], c[0], p_on_element, &result)
    return np.complex64(result.re+result.im*1j)


cpdef n_3d(float k, float[:] p, float[:] vec_p, float[:] qa, float[:] qb, float[:] qc, bool p_on_element):
    cdef Complex result
    cdef Float3 *cp = <Float3*>&p[0]
    cdef Float3 *a = <Float3*>&qa[0]
    cdef Float3 *b = <Float3*>&qb[0]
    cdef Float3 *c = <Float3*>&qc[0]
    cdef Float3 *vp = <Float3*>&vec_p[0]
    ComputeN_3D(k, cp[0], vp[0], a[0], b[0], c[0], p_on_element, &result)
    return np.complex64(result.re+result.im*1j)


# -----------------------------------------------------------------------------
# RAD
# -----------------------------------------------------------------------------
cpdef l_rad(float k, float[:] p, float[:] qa, float[:] qb, p_on_element):
    cdef Complex result
    cdef Float2 *cp = <Float2*>&p[0]
    cdef Float2 *a = <Float2*>&qa[0]
    cdef Float2 *b = <Float2*>&qb[0]
    ComputeL_RAD(k, cp[0], a[0], b[0], p_on_element, &result)
    return np.complex64(result.re+result.im*1j)


cpdef m_rad(float k, float[:] p, float[:] qa, float[:] qb, bool p_on_element):
    cdef Complex result
    cdef Float2 *cp = <Float2*>&p[0]
    cdef Float2 *a = <Float2*>&qa[0]
    cdef Float2 *b = <Float2*>&qb[0]
    ComputeM_RAD(k, cp[0], a[0], b[0], p_on_element, &result)
    return np.complex64(result.re+result.im*1j)


cpdef mt_rad(float k, float[:] p, float[:] vec_p, float[:] qa, float[:] qb, bool p_on_element):
    cdef Complex result
    cdef Float2 *cp = <Float2*>&p[0]
    cdef Float2 *c_normal_p = <Float2*>&vec_p[0]
    cdef Float2 *a = <Float2*>&qa[0]
    cdef Float2 *b = <Float2*>&qb[0]
    ComputeMt_RAD(k, cp[0], c_normal_p[0], a[0], b[0], p_on_element, &result)
    return np.complex64(result.re+result.im*1j)


cpdef n_rad(float k, float[:] p, float[:] vec_p, float[:] qa, float[:] qb, bool p_on_element):
    cdef Complex result
    cdef Float2 *cp = <Float2*>&p[0]
    cdef Float2 *c_normal_p = <Float2*>&vec_p[0]
    cdef Float2 *a = <Float2*>&qa[0]
    cdef Float2 *b = <Float2*>&qb[0]
    ComputeN_RAD(k, cp[0], c_normal_p[0], a[0], b[0], p_on_element, &result)
    return np.complex64(result.re+result.im*1j)
