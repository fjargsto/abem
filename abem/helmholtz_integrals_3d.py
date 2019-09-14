import numpy as np
from numpy.linalg import norm
from .normals import normal_3d


def complex_quad(func, a, b, c):
    samples = np.array([[0.333333333333333, 0.333333333333333, 0.225000000000000],                                                 
                        [0.797426985353087, 0.101286507323456, 0.125939180544827],                                                 
                        [0.101286507323456, 0.797426985353087, 0.125939180544827],                                                 
                        [0.101286507323456, 0.101286507323456, 0.125939180544827],                                                 
                        [0.470142064105115, 0.470142064105115, 0.132394152788506],                                                 
                        [0.470142064105115, 0.059715871789770, 0.132394152788506],                                                 
                        [0.059715871789770, 0.470142064105115, 0.132394152788506]],
                       dtype=np.float32)
                                                                                                                               
    deltaB = b - a                                                                                                                 
    deltaC = c - a                                                                                                                 
    sum = 0.0                                                                                                                      
    for n in range(samples.shape[0]):                                                                                              
        x = a + samples[n, 0] * deltaB + samples[n, 1] * deltaC                                                                    
        sum += samples[n, 2] * func(x)                                                                                             
    return sum * 0.5 * norm(np.cross(deltaB, deltaC))                                                                              


def compute_l(k, p, qa, qb, qc, p_on_element):
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
            L0 = compute_l(0.0, p, qa, qb, qc, True)
            Lk = complex_quad(func, qa, qb, p) + complex_quad(func, qb, qc, p) \
                 + complex_quad(func, qc, qa, p)
            return L0 + Lk / (4.0 * np.pi)                                                                                         
    else:                                                                                                                          
        if k == 0.0:                                                                                                               
            def func(x):                                                                                                           
                r = p - x                                                                                                          
                R = norm(r)                                                                                                        
                return 1.0 / R                                                                                                     
            return complex_quad(func, qa, qb, qc) / (4.0 * np.pi)
        else:                                                                                                                      
            def func(x):                                                                                                           
                r = p - x                                                                                                          
                R = norm(r)                                                                                                        
                ikr = 1j * k * R                                                                                                   
                return np.exp(ikr) / R                                                                                             
            return complex_quad(func, qa, qb, qc) / (4.0 * np.pi)


def compute_m(k, p, qa, qb, qc, p_on_element):
    if p_on_element:
        return 0.0                                                                                                                 
    else:                                                                                                                          
        if k == 0.0:                                                                                                               
            def func(x):                                                                                                           
                r = p - x                                                                                                          
                R = norm(r)                                                                                                        
                rnq = -np.dot(r, normal_3d(qa, qb, qc)) / R
                return -1.0 / np.dot(r, r) * rnq                                                                                   
            return complex_quad(func, qa, qb, qc) / (4.0 * np.pi)
        else:                                                                                                                      
            def func(x):                                                                                                           
                r = p - x                                                                                                          
                R = norm(r)                                                                                                        
                rnq = -np.dot(r, normal_3d(qa, qb, qc)) / R
                kr = k * R                                                                                                         
                ikr = 1j * kr                                                                                                      
                return rnq * (ikr - 1.0) * np.exp(ikr) / np.dot(r, r)                                                              
            return complex_quad(func, qa, qb, qc) / (4.0 * np.pi)

                                                                                                                               
def compute_mt(k, p, vecp, qa, qb, qc, p_on_element):
    if p_on_element:
        return 0.0                                                                                                                 
    else:                                                                                                                          
        if k == 0.0:                                                                                                               
            def func(x):                                                                                                           
                r = p - x                                                                                                          
                R = norm(r)                                                                                                        
                rnp = np.dot(r, vecp) / R                                                                                          
                return -1.0 / np.dot(r, r) * rnp                                                                                   
            return complex_quad(func, qa, qb, qc) / (4.0 * np.pi)
        else:                                                                                                                      
            def func(x):                                                                                                           
                r = p - x                                                                                                          
                R = norm(r)                                                                                                        
                rnp = np.dot(r, vecp) / R                                                                                          
                ikr = 1j * k * R
                return rnp * (ikr - 1.0) * np.exp(ikr) / np.dot(r, r)                                                              
            return complex_quad(func, qa, qb, qc) / (4.0 * np.pi)

                                                                                                                               
def compute_n(k, p, vecp, qa, qb, qc, p_on_element):
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
            N0 = compute_n(0.0, p, vecp, qa, qb, qc, True)
            L0 = compute_l(0.0, p, qa, qb, qc, True)
            Nk = complex_quad(func, qa, qb, p) + complex_quad(func, qb, qc, p) + complex_quad(func, qc, qa, p)
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
                                                                                                                               
            return complex_quad(func, qa, qb, qc) / (4.0 * np.pi)
