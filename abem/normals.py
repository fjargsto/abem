import numpy as np
from numpy.linalg import norm


def normal_2d(a, b):
    diff = a - b
    length = norm(diff)
    return np.array([diff[1]/length, -diff[0]/length])

def normal_3d(a, b, c):
    ab = b - a
    ac = c - a
    normal = np.cross(ab, ac)
    normal /= norm(normal)
    return normal
