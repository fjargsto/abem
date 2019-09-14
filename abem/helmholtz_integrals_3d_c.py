from .native_interface import Float3, Complex, intops
from ctypes import c_bool, c_float, byref
import numpy as np


def compute_l(k, p, qa, qb, qc, p_on_element):
    result = Complex()
    p = Float3(p[0],   p[1],  p[2])                                            
    a = Float3(qa[0], qa[1], qa[2])                                            
    b = Float3(qb[0], qb[1], qb[2])                                            
    c = Float3(qc[0], qc[1], qc[2])                                            
    on = c_bool(p_on_element)
    intops.ComputeL_3D(c_float(k), p, a, b, c, on, byref(result))
    return np.complex64(result.re+result.im*1j)                                


def compute_m(k, p, qa, qb, qc, p_on_element):
    result = Complex()
    p = Float3(p[0], p[1], p[2])                                               
    a = Float3(qa[0], qa[1], qa[2])                                            
    b = Float3(qb[0], qb[1], qb[2])                                            
    c = Float3(qc[0], qc[1], qc[2])                                            
    on = c_bool(p_on_element)
    intops.ComputeM_3D(c_float(k), p, a, b, c, on, byref(result))
    return np.complex64(result.re+result.im*1j)                                


def compute_mt(k, p, vec_p, qa, qb, qc, p_on_element):
    result = Complex()
    p = Float3(p[0], p[1], p[2])                                               
    vp = Float3(vec_p[0], vec_p[1], vec_p[2])                                  
    a = Float3(qa[0], qa[1], qa[2])                                            
    b = Float3(qb[0], qb[1], qb[2])                                            
    c = Float3(qc[0], qc[1], qc[2])                                            
    on = c_bool(p_on_element)
    intops.ComputeMt_3D(c_float(k), p, vp, a, b, c, on, byref(result))
    return np.complex64(result.re+result.im*1j)                                


def compute_n(k, p, vec_p, qa, qb, qc, p_on_element):
    result = Complex()
    p = Float3(p[0], p[1], p[2])                                               
    vp = Float3(vec_p[0], vec_p[1], vec_p[2])                                  
    a = Float3(qa[0], qa[1], qa[2])                                            
    b = Float3(qb[0], qb[1], qb[2])                                            
    c = Float3(qc[0], qc[1], qc[2])                                            
    on = c_bool(p_on_element)
    intops.ComputeN_3D(c_float(k), p, vp, a, b, c, on, byref(result))
    return np.complex64(result.re+result.im*1j)                                
