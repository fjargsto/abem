cdef extern from "stdbool.h":
    ctypedef bint bool

cdef extern from "iops_cpp.h":
    cdef cppclass Float2:
        float x
        float y

    Float2 operator+(Float2, Float2)

    ctypedef struct Float3:
        float x
        float y
        float z

    Float2 add2f(Float2 a, Float2 b)
    Float3 add3f(Float3 a, Float3 b)

    Float2 sub2f(Float2 a, Float2 b)
    Float3 sub3f(Float3 a, Float3 b)

    Float2 smul2f(float a, Float2 x)
    Float3 smul3f(float a, Float3 x)

    float dot2f(Float2 a, Float2 b)
    float dot3f(Float3 a, Float3 b)

    Float3 cross(Float3 a, Float3 b)

    float norm2f(Float2 a)
    float norm3f(Float3 a)

    Float2 Normal2D(Float2 a, Float2 b)
    Float3 Normal3D(Float3 a, Float3 b, Float3 c)

    Float2 add2f(Float2 a, Float2 b)

    ctypedef struct Complex:
        float re
        float im

    void Hankel1(int order, float x, Complex * pz)

    void ComputeL_2D(float k, Float2 p, Float2 a, Float2 b, bool pOnElement, Complex * pResult)
    void ComputeM_2D(float k, Float2 p, Float2 a, Float2 b, bool pOnElement, Complex * pResult)
    void ComputeMt_2D(float k, Float2 p, Float2 normal_p, Float2 a, Float2 b, bool pOnElement, Complex * pResult)
    void ComputeN_2D(float k, Float2 p, Float2 normal_p, Float2 a, Float2 b, bool pOnElement, Complex * pResult)

    void ComputeL_RAD(float k, Float2 p, Float2 a, Float2 b, bool pOnElement, Complex * pResult)
    void ComputeM_RAD(float k, Float2 p, Float2 a, Float2 b, bool pOnElement, Complex * pResult)
    void ComputeMt_RAD(float k, Float2 p, Float2 vec_p, Float2 a, Float2 b, bool pOnElement, Complex * pResult)
    void ComputeN_RAD(float k, Float2 p, Float2 vec_p, Float2 a, Float2 b, bool pOnElement, Complex * pResult)

    void ComputeL_3D(float k, Float3 p, Float3 a, Float3 b, Float3 c, bool pOnElement, Complex * pResult)
    void ComputeM_3D(float k, Float3 p, Float3 a, Float3 b, Float3 c, bool pOnElement, Complex * pResult)
    void ComputeMt_3D(float k, Float3 p, Float3 vec_p, Float3 a, Float3 b, Float3 c, bool pOnElement, Complex * pResult)
    void ComputeN_3D(float k, Float3 p, Float3 normal_p, Float3 a, Float3 b, Float3 c, bool pOnElement, Complex * pResult)
