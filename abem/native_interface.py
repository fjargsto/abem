from ctypes import CDLL, c_int, c_float, c_void_p, Structure


intops = CDLL(r"C:\Users\frank\BoundaryElementMethod\abem\x64\Debug\intops")
intops.Hankel1.argtypes = [c_int, c_float, c_void_p]


class Complex(Structure):
    _fields_ = [("re", c_float), ("im", c_float)]


class Float2(Structure):
    _fields_ = [("x", c_float), ("y", c_float)]


class Float3(Structure):
    _fields_ = [("x", c_float), ("y", c_float), ("z", c_float)]
