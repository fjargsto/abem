import numpy as np
from numpy.linalg import norm
from scipy.special import hankel1
from .normals import normal_2d


def complex_quad(func, start, end):
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
        x = start + samples[n, 0] * vec                                                                                
        sum += samples[n, 1] * func(x)                                                                                 
    return sum * norm(vec)                                                                                         


def compute_l(k, p, qa, qb, p_on_element):
    qab = qb - qa
    if p_on_element:
        if k == 0.0:                                                                                               
            ra = norm(p - qa)                                                                                      
            rb = norm(p - qb)                                                                                      
            re = norm(qab)                                                                                         
            return 0.5 / np.pi * (re - (ra * np.log(ra) + rb * np.log(rb)))                                        
        else:                                                                                                      
            def func(x):                                                                                           
                R = norm(p - x)                                                                                    
                return 0.5 / np.pi * np.log(R) + 0.25j * hankel1(0, k * R)                                         
            return complex_quad(func, qa, p) + complex_quad(func, p, qa) \
                   + compute_l(0.0, p, qa, qb, True)
    else:                                                                                                          
        if k == 0.0:                                                                                               
            return -0.5 / np.pi * complex_quad(lambda q: np.log(norm(p - q)), qa, qb)
        else:                                                                                                      
            return 0.25j * complex_quad(lambda q: hankel1(0, k * norm(p - q)), qa, qb)


def compute_m(k, p, qa, qb, p_on_element):
    vecq = normal_2d(qa, qb)
    if p_on_element:
        return 0.0                                                                                                 
    else:                                                                                                          
        if k == 0.0:                                                                                               
            def func(x):                                                                                           
                r = p - x                                                                                          
                return np.dot(r, vecq) / np.dot(r, r)                                                              
            return -0.5 / np.pi * complex_quad(func, qa, qb)
        else:                                                                                                      
            def func(x):                                                                                           
                r = p - x                                                                                          
                R = norm(r)                                                                                        
                return hankel1(1, k * R) * np.dot(r, vecq) / R                                                     
            return 0.25j * k * complex_quad(func, qa, qb)

                                                                                                                   
def compute_mt(k, p, vecp, qa, qb, p_on_element):
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


def compute_n(k, p, vecp, qa, qb, p_on_element):
    qab = qb - qa
    if p_on_element:
        ra = norm(p - qa)                                                                                          
        rb = norm(p - qb)                                                                                          
        re = norm(qab)                                                                                             
        if k == 0.0:                                                                                               
            return -(1.0 / ra + 1.0 / rb) / (re * 2.0 * np.pi) * re                                                
        else:                                                                                                      
            vecq = normal_2d(qa, qb)
            k2 = k * k

            def func(x):                                                                                           
                r = p - x                                                                                          
                R2 = np.dot(r, r)                                                                                  
                R = np.sqrt(R2)                                                                                    
                drdudrdn = -np.dot(r, vecq) * np.dot(r, vecp) / R2                                                 
                dpnu = np.dot(vecp, vecq)                                                                          
                c1 = 0.25j * k / R * hankel1(1, k * R) - 0.5 / (np.pi * R2)
                c2 = 0.50j * k / R * hankel1(1, k * R) - 0.25j * k2 * \
                    hankel1(0, k * R) - 1.0 / (np.pi * R2)
                c3 = -0.25 * k2 * np.log(R) / np.pi
                return c1 * dpnu + c2 * drdudrdn + c3

            return compute_n(0.0, p, vecp, qa, qb, True) - 0.5 * k2 * \
                compute_l(0.0, p, qa, qb, True) + \
                complex_quad(func, qa, p) + complex_quad(func, p, qb)
    else:                                                                                                          
        sum = 0.0j                                                                                                 
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
                return hankel1(1, k * R) / R * (un + 2.0 * drdudrdn) - k * hankel1(0, k * R) * drdudrdn            
            return 0.25j * k * complex_quad(func, qa, qb)
