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

    # -------------------------------------------------------------------------
    # 2D Integral Operators
    #
    void L_2D_ON_K0(const Float2 *pa, const Float2 *pb, Complex * pResult)
    void L_2D_ON(float k, const Float2 *pp, const Float2 *pa, const Float2 *pb, Complex * pResult)
    void L_2D_OFF_K0(const Float2 *pp, const Float2 *pa, const Float2 *pb, Complex * pResult)
    void L_2D_OFF(float k, const Float2* pp, const Float2* pa, const Float2* pb, Complex* pResult)
    void L_2D(float k, const Float2 *pp, const Float2 *pa, const Float2 *pb, bool pOnElement, Complex * pResult)

    void M_2D_OFF_K0(const Float2* pp, const Float2* pa, const Float2* pb, Complex * pResult)
    void M_2D_OFF(float k, const Float2* pp, const Float2* pa, const Float2* pb, Complex * pResult)
    void M_2D(float k, const Float2 *pp, const Float2 *pa, const Float2 *pb, bool pOnElement, Complex * pResult)

    void MT_2D_OFF_K0(const Float2* p, const Float2* normal_p, const Float2* a, const Float2* b, Complex * pResult)
    void MT_2D_OFF(float k, const Float2* p, const Float2* normal_p, const Float2* a, const Float2* b, Complex * pResult)
    void MT_2D(float k, const Float2 *pp, const Float2 *p_normal_p, const Float2 *pa, const Float2 *pb, bool pOnElement,
               Complex * pResult)

    void N_2D_ON_K0(const Float2* a, const Float2* b, Complex* pResult)
    void N_2D_ON(float k, const Float2* p, const Float2* normal_p, const Float2* a, const Float2* b, Complex * pResult)
    void N_2D_OFF_K0(const Float2* p, const Float2* normal_p, const Float2* a, const Float2* b, Complex * pResult)
    void N_2D_OFF(float k, const Float2* p, const Float2* normal_p, const Float2* a, const Float2* b, Complex * pResult)
    void N_2D(float k, const Float2* pp, const Float2* p_normal_p, const Float2* pa, const Float2* pb, bool pOnElement,
              Complex * pResult)

    # -------------------------------------------------------------------------
    # 2D Boundary Matrices
    #
    ctypedef struct LineSegment:
        Float2 a
        Float2 b

    void BOUNDARY_MATRICES_2D(float k, const Complex* p_mu, const LineSegment* p_edges, Complex* p_a, Complex* p_b,
                              unsigned int N, float orientation)  nogil except +
    void SOLUTION_MATRICES_2D(float k, const Float2* p_samples, const LineSegment* p_edges,
                              Complex* p_l, Complex* p_m,
                              unsigned int N, unsigned int M) nogil except +
    void SAMPLE_PHI_2D(float k, const Float2* p_samples, const LineSegment* viewEdges, unsigned int N, unsigned int M,
                       const Complex* p_solution_phi, const Complex* p_solution_v,  Complex* p_phi) nogil except +


    # -------------------------------------------------------------------------
    # Radial Integral Operators
    #
    void ComputeL_RAD(float k, Float2 p, Float2 a, Float2 b, bool pOnElement, Complex * pResult)
    void ComputeM_RAD(float k, Float2 p, Float2 a, Float2 b, bool pOnElement, Complex * pResult)
    void ComputeMt_RAD(float k, Float2 p, Float2 vec_p, Float2 a, Float2 b, bool pOnElement, Complex * pResult)
    void ComputeN_RAD(float k, Float2 p, Float2 vec_p, Float2 a, Float2 b, bool pOnElement, Complex * pResult)

    void ComputeL_3D(float k, Float3 p, Float3 a, Float3 b, Float3 c, bool pOnElement, Complex * pResult)
    void ComputeM_3D(float k, Float3 p, Float3 a, Float3 b, Float3 c, bool pOnElement, Complex * pResult)
    void ComputeMt_3D(float k, Float3 p, Float3 vec_p, Float3 a, Float3 b, Float3 c, bool pOnElement, Complex * pResult)
    void ComputeN_3D(float k, Float3 p, Float3 normal_p, Float3 a, Float3 b, Float3 c, bool pOnElement, Complex * pResult)
