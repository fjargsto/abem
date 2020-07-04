/* --------------------------------------------------------------------------- *
 * Copyright (C) 2017, 2020 Frank Jargstorff                                   *
 *                                                                             *
 * This file is part of the AcousticBEM library.                               *
 * AcousticBEM is free software: you can redistribute it and/or modify         *
 * it under the terms of the GNU General Public License as published by        *
 * the Free Software Foundation, either version 3 of the License, or           *
 * (at your option) any later version.                                         *
 *                                                                             *
 * AcousticBEM is distributed in the hope that it will be useful,              *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of              *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               *
 * GNU General Public License for more details.                                *
 *                                                                             *
 * You should have received a copy of the GNU General Public License           *
 * along with AcousticBEM.  If not, see <http://www.gnu.org/licenses/>.        *
 * --------------------------------------------------------------------------- */
#include "iops_cpp.h"

#include <cassert>
#include <iostream>

#ifdef _MSC_VER
inline float jnf(int order, float x) {return (float)_jn(order, (double)x);}
inline float ynf(int order, float x) {return (float)_yn(order, (double)x);}
#endif

#define MAX_LINE_RULE_SAMPLES 4096

using namespace std::complex_literals;


/* ********************************************************************************************** */


float aX_1D[] = {0.980144928249f,     0.898333238707f, 0.762766204958f, 0.591717321248f,
		 0.408282678752f,     0.237233795042f, 0.101666761293f, 1.985507175123E-02f};
float aW_1D[] = {5.061426814519E-02f, 0.111190517227f, 0.156853322939f, 0.181341891689f,
		 0.181341891689f,     0.156853322939f, 0.111190517227f, 5.061426814519E-02f};

float aX_2D[] = {0.333333333333f,     0.797426985353f, 0.101286507323f, 0.101286507323f,
		 0.470142064105f,     0.470142064105f, 0.059715871789f};
float aY_2D[] = {0.333333333333f,     0.101286507323f, 0.797426985353f, 0.101286507323f,
		 0.470142064105f,     0.059715871789f, 0.470142064105f};
float aW_2D[] = {0.225000000000f,     0.125939180544f, 0.125939180544f, 0.125939180544f,
		 0.132394152788f,     0.132394152788f, 0.132394152788f};

std::complex<float> hankel1(int order, float x) {
  return std::complex<float>(jnf(order, x), ynf(order, x));
}

void Hankel1(int order, float x, Complex * pz) {
  std::complex<float> z = hankel1(order, x);
  pz->re = z.real();
  pz->im = z.imag();
}

struct GaussLegendre8
{
  static const unsigned int N = 4;
  static const constexpr float x[] = {0.18343464249600006f, 0.5255324099159999f, 0.796666477414f, 0.960289856498f  };
  static const constexpr float w[] = {0.362683783378f,      0.313706645878f,     0.222381034454f, 0.10122853629038f};
};

const unsigned int    GaussLegendre8::N;
const constexpr float GaussLegendre8::x[];
const constexpr float GaussLegendre8::w[];

struct GaussLegendre10
{
  static const unsigned int N = 5;
  static const constexpr float x[] = {0.1488743389, 0.4333953941, 0.6794095682, 0.8650633666, 0.9739065285};
  static const constexpr float w[] = {0.2955242247, 0.2692667193, 0.2190863625, 0.1494513491, 0.0666713443};
};

const unsigned int    GaussLegendre10::N;
const constexpr float GaussLegendre10::x[];
const constexpr float GaussLegendre10::w[];

template<typename W>
class GaussianQuadrature
{
public:
  template<class F> // a functor: R -> C
  std::complex<float> operator()(F func, float a, float b) {
    const float xm = 0.5f * (b + a);
    const float xr = 0.5f * (b - a);
    std::complex<float> sum = 0.0f + 0.0fj;
    for (unsigned int j = 0; j < W::N; ++j) {
      const float dx = xr * W::x[j];
      sum += W::w[j] * (func(xm + dx) + func(xm - dx));
    }
    return sum *= xr;
  }
};

Float2 lerp(float t, const Float2 &a, const Float2 &b) {
  return a + t * (b - a);
}

std::complex<float> complexQuad2D(std::complex<float> (*integrand)(const Float2*, void*), void * state, IntRule1D intRule,
			    const Float2 *p_start, const Float2 *p_end) {
  Float2 vec;
  vec = *p_end - *p_start;
  std::complex<float> sum = 0.0f;
  for (int i = 0; i < intRule.nSamples; ++i) {
    Float2 x = intRule.pX[i] * vec + *p_start;
    sum += intRule.pW[i] * integrand(&x, state);
  }

  return norm(vec) * sum;
}

std::complex<float> complexQuadGenerator(std::complex<float> (*integrand)(const Float2*, void*), void * state, IntRule1D intRule,
				   Float2 start, Float2 end) {
  Float2 vec;
  vec = end - start;
  std::complex<float> sum = 0.0f;
  for (int i = 0; i < intRule.nSamples; ++i) {
    float x_sqr = intRule.pX[i] * intRule.pX[i];
    Float2 x = x_sqr * vec + start;
    sum += intRule.pW[i] * integrand(&x, state) * intRule.pX[i];
  }
  return norm(vec) * sum * 2.0f;
}

std::complex<float> complexQuad3D(std::complex<float>(*integrand)(Float3, void*), void * state, IntRule2D intRule,
			    Float3 a, Float3 b, Float3 c) {
  Float3 vec_b = b - a;
  Float3 vec_c = c - a;
  std::complex<float> sum = 0.0f;
  for (int i = 0; i < intRule.nSamples; ++i) {
    Float3 x = (a + (intRule.pX[i] * vec_b));
    x = (x + (intRule.pY[i] * vec_c));
    sum += intRule.pW[i] * integrand(x, state);
  }
  return 0.5f * norm(cross(vec_b, vec_c)) * sum;
}

void semiCircleIntegralRule(int nSections, IntRule1D intRule, IntRule2D * pSemiCircleRule) {
  pSemiCircleRule->nSamples = nSections * intRule.nSamples;

  float factor = M_PI / nSections;
  for (int i = 0; i < pSemiCircleRule->nSamples; ++i) {
    float arcAbscissa = (i / intRule.nSamples + intRule.pX[i % intRule.nSamples]) * factor;
    pSemiCircleRule->pX[i] = cosf(arcAbscissa);
    pSemiCircleRule->pY[i] = sinf(arcAbscissa);
    pSemiCircleRule->pW[i] = intRule.pW[i % intRule.nSamples] * factor;
  }
}

std::complex<float> complexLineIntegral(std::complex<float> (*integrand)(Float2, void*), void * state,
				  IntRule2D intRule) {
  std::complex<float> sum = 0.0f;
  for (int i = 0; i < intRule.nSamples; ++i) {
    Float2 x = {intRule.pX[i], intRule.pY[i]};
    sum += intRule.pW[i] * integrand(x, state);
  }
  return sum;
}

// -----------------------------------------------------------------------------
// 2D discrete Helmholtz operators.
// -----------------------------------------------------------------------------

typedef struct {
  float k;
  const Float2 *pp;
  const Float2 *p_normal_p;
  const Float2 *p_normal_q;
} IntL;

std::complex<float> l_2d_on_k0(const Float2 *pa, const Float2 *pb) {
  float l = norm(*pb - *pa);

  return 0.5f * M_1_PI * l * (1.0f - logf(0.5f * l));
}

void L_2D_ON_K0(const Float2 *pa, const Float2 *pb, Complex * pResult) {
  std::complex<float> z;

  z = l_2d_on_k0(pa, pb);
  
  pResult->re = z.real();
  pResult->im = z.imag();
}

class IntL2d_on
{
public:
  IntL2d_on(const float k, const Float2 *pp, const Float2 *pa, const Float2 *pb): k_(k)
										, pp_(pp)
										, pa_(pa)
										, pb_(pb)
  { ; }

  std::complex<float> operator()(float t) {
    Float2 q = lerp(t, *pa_, *pb_);
    float r = norm(*pp_ - q);

    return static_cast<float>(0.5f * M_1_PI * logf(r)) + 0.25f * 1if * hankel1(0.0f, k_ * r);
  }

private:
  const float k_;
  const Float2 *pp_;
  const Float2 *pa_;
  const Float2 *pb_;
};

std::complex<float> l_2d_on(float k, const Float2 *pp, const Float2 *pa, const Float2 *pb) {
  GaussianQuadrature<GaussLegendre8> integrate;
  float l = norm(*pa - *pb);
  
  return (integrate(IntL2d_on(k, pp, pa, pp), 0.0f, 1.0f) + integrate(IntL2d_on(k, pp, pp, pb), 0.0f, 1.0f)) * 0.5f * l
    + l_2d_on_k0(pa, pb);
}

void L_2D_ON(float k, const Float2 *pp, const Float2 *pa, const Float2 *pb, Complex * pResult) {
  std::complex<float> z;

  z = l_2d_on(k, pp, pa, pb);
  
  pResult->re = z.real();
  pResult->im = z.imag();
}

class IntL2d_off_k0
{
public:
  IntL2d_off_k0(const Float2 *pp, const Float2 *pa, const Float2 *pb): pp_(pp)
								     , pa_(pa)
								     , pb_(pb)
  { ; }

  std::complex<float> operator()(float t) {
    Float2 q = lerp(t, *pa_, *pb_);
    float r = norm(*pp_ - q);

    return log(r);
  }

private:
  const Float2 *pp_;
  const Float2 *pa_;
  const Float2 *pb_;
};

std::complex<float> l_2d_off_k0(const Float2 *pp, const Float2 *pa, const Float2 *pb) {
  GaussianQuadrature<GaussLegendre8> integrate;
  float l = norm(*pa - *pb);

  return static_cast<float>(-0.5 * M_1_PI) * integrate(IntL2d_off_k0(pp, pa, pb), 0.0f, 1.0f) * l;
}

void L_2D_OFF_K0(const Float2 *pp, const Float2 *pa, const Float2 *pb, Complex * pResult) {
  std::complex<float> z;

  z = l_2d_off_k0(pp, pa, pb);
  
  pResult->re = z.real();
  pResult->im = z.imag();
}

class IntL2d_off
{
public:
  IntL2d_off(const float k, const Float2* pp, const Float2* pa, const Float2* pb): k_(k)
										 , pp_(pp)
										 , pa_(pa)
										 , pb_(pb)
  { ; }

  std::complex<float> operator()(float t) {
    Float2 q = lerp(t, *pa_, *pb_);
    float r = norm(*pp_ - q);

    return hankel1(0, k_ * r);
  }

private:
  const float k_;
  const Float2 *pp_;
  const Float2 *pa_;
  const Float2 *pb_;
};

std::complex<float> l_2d_off(float k, const Float2* pp, const Float2* pa, const Float2* pb) {
  GaussianQuadrature<GaussLegendre8> integrate;
  float l = norm(*pa - *pb);
  
  return 0.25f * 1if * integrate(IntL2d_off(k, pp, pa, pb), 0.0f, 1.0f) * l;
}

void L_2D_OFF(float k, const Float2* pp, const Float2* pa, const Float2* pb, Complex* pResult) {
  std::complex<float> z;

  z = l_2d_off(k, pp, pa, pb);
  
  pResult->re = z.real();
  pResult->im = z.imag();
}

void L_2D(float k, const Float2* pp, const Float2* pa, const Float2* pb, bool pOnElement, Complex* pResult) {
  std::complex<float> z;
  if (pOnElement) {
    if (k == 0.0f)
      z = l_2d_on_k0(pa, pb);
    else
      z = l_2d_on(k, pp, pa, pb);
  } else {
    if (k == 0.0f)
      z = l_2d_off_k0(pp, pa, pb);
    else
      z = l_2d_off(k, pp, pa, pb);
  }
  pResult->re = z.real();
  pResult->im = z.imag();
}

class IntM2d_off_k0
{
public:
  IntM2d_off_k0(const Float2 *pp, const Float2 *pa, const Float2 *pb, const Float2 *pnq): pp_(pp)
											, pa_(pa)
											, pb_(pb)
											, pnq_(pnq)
  { ; }

  std::complex<float> operator()(float t) {
    Float2 q = lerp(t, *pa_, *pb_);
    Float2 r = *pp_ - q;

    return dot(r,*pnq_) / dot(r, r);
  }

private:
  const Float2 *pp_;
  const Float2 *pa_;
  const Float2 *pb_;
  const Float2 *pnq_;
};

std::complex<float> m_2d_off_k0(const Float2 *pp, const Float2 *pa, const Float2 *pb) {
  GaussianQuadrature<GaussLegendre8> integrate;
  float l = norm(*pa - *pb);
  Float2 n_q = normal(*pa, *pb);

  return static_cast<float>(0.5f * M_1_PI) * integrate(IntM2d_off_k0(pp, pa, pb, &n_q), 0.0f, 1.0f) * l;
}

void M_2D_OFF_K0(const Float2 *pp, const Float2 *pa, const Float2 *pb, Complex * pResult) {
  std::complex<float> z;

  z = m_2d_off_k0(pp, pa, pb);
  
  pResult->re = z.real();
  pResult->im = z.imag();
}

std::complex<float> int_m_2d_off(const Float2 *px, void* state) {
  IntL * s = (IntL *) state;
  Float2 r = *s->pp - *px;
  float R = norm(r);
  return hankel1(1, s->k * R) * dot(r, *s->p_normal_q) / R;
}

class IntM2d_off
{
public:
  IntM2d_off(const float k, const Float2 *pp, const Float2 *pa, const Float2 *pb, const Float2 *pnq): k_(k)
												    , pp_(pp)
												    , pa_(pa)
												    , pb_(pb)
												    , pnq_(pnq)
  { ; }

  std::complex<float> operator()(float t) {
    Float2 q = lerp(t, *pa_, *pb_);
    Float2 r = *pp_ - q;
    float R = norm(r);

    return hankel1(1, k_ * R) * dot(r, *pnq_) / R;
  }

private:
  const float k_;
  const Float2 *pp_;
  const Float2 *pa_;
  const Float2 *pb_;
  const Float2 *pnq_;
};

std::complex<float> m_2d_off(float k, const Float2 *pp, const Float2 *pa, const Float2 *pb) {
  GaussianQuadrature<GaussLegendre8> integrate;
  float l = norm(*pa - *pb);
  Float2 n_q = normal(*pa, *pb);
  
  return 0.25f * 1if * k * integrate(IntM2d_off(k, pp, pa, pb, &n_q), 0.0f, 1.0f) * l;
}

void M_2D_OFF(float k, const Float2 *pp, const Float2 *pa, const Float2 *pb, Complex * pResult) {
  std::complex<float> z;

  z = m_2d_off(k, pp, pa, pb);
  
  pResult->re = z.real();
  pResult->im = z.imag();
}

void M_2D(float k, const Float2 *pp, const Float2 *pa, const Float2 *pb, bool pOnElement, Complex * pResult) {
  std::complex<float> z;
  if (pOnElement)
    z = 0.0;
  else {
    if (k == 0.0f)
      z = m_2d_off_k0(pp, pa, pb);
    else
      z = m_2d_off(k, pp, pa, pb);
  }
  pResult->re = z.real();
  pResult->im = z.imag();
}

std::complex<float> mt_2d_off_k0(const Float2 *pp, const Float2 *pnp, const Float2 *pa, const Float2 *pb) {
  GaussianQuadrature<GaussLegendre8> integrate;
  float l = norm(*pa - *pb);

  return static_cast<float>(-0.5f * M_1_PI) * integrate(IntM2d_off_k0(pp, pa, pb, pnp), 0.0f, 1.0f) * l;
}

void MT_2D_OFF_K0(const Float2 *pp, const Float2 *p_normal_p, const Float2 *pa, const Float2 *pb, Complex * pResult) {
  std::complex<float> z;

  z = mt_2d_off_k0(pp, p_normal_p, pa, pb);

  pResult->re = z.real();
  pResult->im = z.imag();
}

std::complex<float> mt_2d_off(float k, const Float2 *pp, const Float2 *pnp, const Float2 *pa, const Float2 *pb) {
  GaussianQuadrature<GaussLegendre8> integrate;
  float l = norm(*pa - *pb);

  return -0.25f * 1if * k * integrate(IntM2d_off(k, pp, pa, pb, pnp), 0.0f, 1.0f) * l;
}

void MT_2D_OFF(float k, const Float2 *pp, const Float2 *p_normal_p, const Float2 *pa, const Float2 *pb, Complex * pResult) {
  std::complex<float> z;

  z = mt_2d_off(k, pp, p_normal_p, pa, pb);

  pResult->re = z.real();
  pResult->im = z.imag();
}

std::complex<float> mt_2d(float k, const Float2 *pp, const Float2 *p_normal_p, const Float2 *pa, const Float2 *pb,
                          bool pOnElement) {
  if (pOnElement)
    return 0.0;
  else {
    if (k == 0.0f)
      return mt_2d_off_k0(pp, p_normal_p, pa, pb);
    else
      return mt_2d_off(k, pp, p_normal_p, pa, pb);
  }
}

void MT_2D(float k, const Float2 *pp, const Float2 *p_normal_p, const Float2 *pa, const Float2 *pb, bool pOnElement,
           Complex * pResult) {
  std::complex<float> z;
  if (pOnElement)
    z = 0.0;
  else {
    if (k == 0.0f)
      z = mt_2d_off_k0(pp, p_normal_p, pa, pb);
    else
      z = mt_2d_off(k, pp, p_normal_p, pa, pb);
  }
  pResult->re = z.real();
  pResult->im = z.imag();
}

std::complex<float> n_2d_on_k0(const Float2 *pa, const Float2 *pb) {
  float ab = norm(*pb - *pa);
  
  return -2.0f / (M_PI * ab);
}

void N_2D_ON_K0(const Float2 *pa, const Float2 *pb, Complex *pResult) {
  std::complex<float> z;

  z = n_2d_on_k0(pa, pb);
  
  pResult->re = z.real();
  pResult->im = z.imag();
}

class IntN2d_on
{
public:
  IntN2d_on(const float k, const Float2 *pp, const Float2 *pnp, const Float2 *pa, const Float2 *pb,
	    const Float2 *pnq): k_(k)
			      , pp_(pp)
			      , pnp_(pnp)
			      , pa_(pa)
			      , pb_(pb)
			      , pnq_(pnq)
  { ; }

  std::complex<float> operator()(float t) {
    Float2 q = lerp(t, *pa_, *pb_);
    Float2 r = *pp_ - q;
    float R2 = dot(r, r);
    float R = sqrtf(R2);
    float drdudrdn = -dot(r, *pnq_) * dot(r, *pnp_) / R2;
    float dpnu = dot(*pnp_, *pnq_);
    std::complex<float> c1 = 0.25f * 1if * k_ / R * hankel1(1, k_ * R) - 0.5f * static_cast<float>(M_1_PI) / R2;
    std::complex<float> c2 = 0.50f * 1if * k_ / R * hankel1(1, k_ * R)
      - 0.25f * 1if * k_ * k_ * hankel1(0, k_ * R) - static_cast<float>(M_1_PI) / R2;
    float c3 = -0.25f * k_ * k_ * logf(R) * static_cast<float>(M_1_PI);
    
    return c1 * dpnu + c2 * drdudrdn + c3;
  }

private:
  const float k_;
  const Float2 *pp_;
  const Float2 *pnp_;
  const Float2 *pa_;
  const Float2 *pb_;
  const Float2 *pnq_;
};

std::complex<float> n_2d_on(float k, const Float2 *pp, const Float2 *pnp, const Float2 *pa, const Float2 *pb) {
  GaussianQuadrature<GaussLegendre8> integrate;
  float l = norm(*pa - *pb);
  Float2 n_q = normal(*pa, *pb);

  return n_2d_on_k0(pa, pb) - 0.5f * k * k * l_2d_on_k0(pa, pb) +
    (integrate(IntN2d_on(k, pp, pnp, pa, pp, &n_q), 0.0, 1.0) + integrate(IntN2d_on(k, pp, pnp, pp, pb, &n_q), 0.0, 1.0)) * 0.5f * l;
}

void N_2D_ON(float k, const Float2 *pp, const Float2 *p_normal_p, const Float2 *pa, const Float2 *pb, Complex *pResult) {
  std::complex<float> z;

  z = n_2d_on(k, pp, p_normal_p, pa, pb);
  
  pResult->re = z.real();
  pResult->im = z.imag();
}

class IntN2d_off_k0
{
public:
  IntN2d_off_k0(const Float2 *pp, const Float2 *pnp, const Float2 *pa, const Float2 *pb,
		const Float2 *pnq): pp_(pp)
				  , pnp_(pnp)
				  , pa_(pa)
				  , pb_(pb)
				  , pnq_(pnq)
  { ; }

  std::complex<float> operator()(float t) {
    Float2 q = lerp(t, *pa_, *pb_);
    Float2 r = *pp_ - q;
    float R2 = dot(r, r);
    float drdudrdn = -dot(r, *pnq_) * dot(r, *pnp_) / R2;
    float dpnu = dot(*pnp_, *pnq_);
    
    return (dpnu + 2.0f * drdudrdn) / R2;
  }

private:
  const Float2 *pp_;
  const Float2 *pnp_;
  const Float2 *pa_;
  const Float2 *pb_;
  const Float2 *pnq_;
};

std::complex<float> n_2d_off_k0(const Float2 *pp, const Float2 *pnp, const Float2 *pa, const Float2 *pb) {
  GaussianQuadrature<GaussLegendre8> integrate;
  float l = norm(*pa - *pb);
  Float2 n_q = normal(*pa, *pb);

  return static_cast<float>(0.5 * M_1_PI) * integrate(IntN2d_off_k0(pp, pnp, pa, pb, &n_q), 0.0f, 1.0f) * l;
}

void N_2D_OFF_K0(const Float2 *pp, const Float2 *p_normal_p, const Float2 *pa, const Float2 *pb, Complex *pResult) {
  std::complex<float> z;

  z = n_2d_off_k0(pp, p_normal_p, pa, pb);
  
  pResult->re = z.real();
  pResult->im = z.imag();
}

class IntN2d_off
{
public:
  IntN2d_off(const float k, const Float2 *pp, const Float2 *pnp, const Float2 *pa, const Float2 *pb,
	     const Float2 *pnq): k_(k)
			       , pp_(pp)
			       , pnp_(pnp)
			       , pa_(pa)
			       , pb_(pb)
			       , pnq_(pnq)
  { ; }

  std::complex<float> operator()(float t) {
    Float2 q = lerp(t, *pa_, *pb_);
    Float2 r = *pp_ - q;
    float R2 = dot(r, r);
    float R = sqrtf(R2);
    float drdudrdn = -dot(r, *pnq_) * dot(r, *pnp_) / R2;
    float dpnu = dot(*pnp_, *pnq_);

  return hankel1(1, k_ * R) / R * (dpnu + 2.0f * drdudrdn)
    - k_ * hankel1(0, k_ * R) * drdudrdn;
  }

private:
  const float k_;
  const Float2 *pp_;
  const Float2 *pnp_;
  const Float2 *pa_;
  const Float2 *pb_;
  const Float2 *pnq_;
};

std::complex<float> n_2d_off(float k, const Float2 *pp, const Float2 *pnp, const Float2 *pa, const Float2 *pb) {
  GaussianQuadrature<GaussLegendre8> integrate;
  float l = norm(*pa - *pb);
  Float2 n_q = normal(*pa, *pb);
  
  return 0.25f * 1if * k * integrate(IntN2d_off(k, pp, pnp, pa, pb, &n_q), 0.0f, 1.0f) * l;
}

void N_2D_OFF(float k, const Float2 *pp, const Float2 *p_normal_p, const Float2 *pa, const Float2 *pb,
              Complex *pResult) {
  std::complex<float> z;

  z = n_2d_off(k, pp, p_normal_p, pa, pb);
  
  pResult->re = z.real();
  pResult->im = z.imag();
}

void N_2D(float k, const Float2 *pp, const Float2 *p_normal_p, const Float2 *pa, const Float2 *pb, bool pOnElement,
          Complex * pResult) {
  std::complex<float> z;
  if (pOnElement) {
    if (k == 0.0f)
      z = n_2d_on_k0(pa, pb);
    else
      z = n_2d_on(k, pp, p_normal_p, pa, pb);
  } else {
    if (k == 0.0f)
      z = n_2d_off_k0(pp, p_normal_p, pa, pb);
    else
      z = n_2d_off(k, pp, p_normal_p, pa, pb);
  }
  pResult->re = z.real();
  pResult->im = z.imag();
}


// -----------------------------------------------------------------------------
// 2D  matrix generators
// -----------------------------------------------------------------------------

void BOUNDARY_MATRICES_2D(float k, const Complex *p_mu_, const LineSegment* p_edges, Complex* p_a_, Complex* p_b_,
			  unsigned int N, float orientation) {
  std::complex<float> mu = *reinterpret_cast<const std::complex<float>*>(p_mu_);
  std::complex<float>* p_a = reinterpret_cast<std::complex<float>*>(p_a_);
  std::complex<float>* p_b = reinterpret_cast<std::complex<float>*>(p_b_);
  #pragma omp parallel for
  for (unsigned int i = 0; i < N; ++i) {
    const LineSegment segment_i = p_edges[i];
    Float2 p = 0.5f * (segment_i.b + segment_i.a);
    Float2 n_p = normal(segment_i.a, segment_i.b);
    for (unsigned int j = 0; j < N; ++j) {
      const LineSegment segment_j = p_edges[j];
      std::complex<float> l, m, mt, n;
      if (i == j) {
	l = l_2d_on(k, &p, &segment_j.a, &segment_j.b);
	m = 0.0f;
	mt = 0.0f;
	n = n_2d_on(k, &p, &n_p, &segment_j.a, &segment_j.b);
      } else {
	l = l_2d_off(k, &p, &segment_j.a, &segment_j.b);
	m = m_2d_off(k, &p, &segment_j.a, &segment_j.b);
	mt = mt_2d_off(k, &p, &n_p, &segment_j.a, &segment_j.b);
	n = n_2d_off(k, &p, &n_p, &segment_j.a, &segment_j.b);
      }
      p_a[i * N + j] = l + mt * mu;
      p_b[i * N + j] = m + n * mu;
    }
    p_a[i * N + i] += orientation * 0.5f * mu;
    p_b[i * N + i] -= orientation * 0.5f;
  }
}

  
void SOLUTION_MATRICES_2D(float k, const Float2* p_samples, const LineSegment* p_edges, Complex* p_l_, Complex* p_m_,
			  unsigned int N, unsigned int M) {
  std::complex<float>* p_l = reinterpret_cast<std::complex<float>*>(p_l_);
  std::complex<float>* p_m = reinterpret_cast<std::complex<float>*>(p_m_);

  #pragma omp parallel for
  for (unsigned int i = 0; i < N; ++i) {
    for (unsigned int j = 0; j < M; ++j) {
      const LineSegment segment_j = p_edges[j];
      p_l[i * M + j] = l_2d_off(k, &p_samples[i], &segment_j.a, &segment_j.b);
      p_m[i * M + j] = m_2d_off(k, &p_samples[i], &segment_j.a, &segment_j.b);
    }
  }
}


void SAMPLE_PHI_2D(float k, const Float2* p_samples, const LineSegment* p_edges, unsigned int N, unsigned int M,
		   const Complex* p_solution_phi, const Complex* p_solution_v,  Complex* p_sample_phi) {
  const std::complex<float>* p_sol_phi = reinterpret_cast<const std::complex<float>*>(p_solution_phi);
  const std::complex<float>* p_sol_v   = reinterpret_cast<const std::complex<float>*>(p_solution_v);
  std::complex<float>* p_phi = reinterpret_cast<std::complex<float>*>(p_sample_phi);

  #pragma omp parallel for
  for (unsigned int i = 0; i < N; ++i) {
    std::complex<float> sum = 0.0f;
    for (unsigned int j = 0; j < M; ++j) {
      const LineSegment segment_j = p_edges[j];
      const std::complex<float> l = l_2d_off(k, &p_samples[i], &segment_j.a, &segment_j.b);
      const std::complex<float> m = m_2d_off(k, &p_samples[i], &segment_j.a, &segment_j.b);
      sum += l * p_sol_v[j] - m * p_sol_phi[j];
    }
    p_phi[i] = sum;
  }
}

/* --------------------------------------------------------------------------  */
/*         Radially symmetrical discrete Helmholtz operators.                  */
/* --------------------------------------------------------------------------- */

typedef struct {
  float k;
  Float3 p;
  Float2 np;
  Float2 nq;

  float r;
  float z;

  IntRule2D semiCircleRule;

  int direction;
} RadIntL;


std::complex<float> integrateSemiCircleL_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;
  Float3 q = {pS->r * x.x, pS->r * x.y, pS->z};
  float R = norm(q - pS->p);

  return std::exp(1if * pS->k * R) / R;
}

std::complex<float> integrateGeneratorL_RAD(const Float2 *px, void *pState) {
  RadIntL * pS = (RadIntL *) pState;

  pS->r = px->x;
  pS->z = px->y;

  return complexLineIntegral(integrateSemiCircleL_RAD, pS, pS->semiCircleRule) * pS->r / static_cast<float>(2.0 * M_PI);
}

std::complex<float> integrateSemiCircleL0_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;
  Float3 q = {pS->r * x.x, pS->r * x.y, pS->z};
  float R = norm(q - pS->p);

  return 1.0f / R;
}

std::complex<float> integrateGeneratorL0_RAD(const Float2 *px, void *pState) {
  RadIntL * pS = (RadIntL *) pState;

  pS->r = px->x;
  pS->z = px->y;

  return complexLineIntegral(integrateSemiCircleL0_RAD, pS, pS->semiCircleRule) * pS->r / static_cast<float>(2.0f * M_PI);
}

std::complex<float> integrateSemiCircleL0pOn_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;
  Float3 q = {pS->r * x.x, pS->r * x.y, pS->z};
  float R = norm(q - pS->p);

  return (std::exp(1if * pS->k * R) - 1.0f) / R;
}

std::complex<float> integrateGeneratorL0pOn_RAD(const Float2 *px, void *pState) {
  RadIntL * pS = (RadIntL *) pState;

  pS->r = px->x;
  pS->z = px->y;

  return complexLineIntegral(integrateSemiCircleL0pOn_RAD, pS, pS->semiCircleRule) * pS->r / static_cast<float>(2.0 * M_PI);
}

/* Computes elements of the Helmholtz L-Operator radially symetrical cases.
 *
 * Parameters:
 *   k - the wavenumber of the problem.
 *   px, py - the point receiving radiation from the boundary.
 *   ar, az - the starting point of the boundary element being integrated over.
 *   br, bz - the end point of the bondary element being integrated over.
 *   pOnElement - a boolean indicating if p is on the boundary element being integrated.
 *
 *   Returns:
 *     The complex-valued result of the integration.
 */
std::complex<float> computeL_RAD(float k, Float2 p, Float2 a, Float2 b, bool pOnElement) {
  IntRule1D intRule = {8, aX_1D, aW_1D};

  /* subdivide circular integral into sections of similar size as qab */
  Float2 q = 0.5f * (a + b);
  Float2 ab = b - a;
  int nSections = 1 + (int)(q.x * M_PI / norm(ab));

  float aSemiCircleX[MAX_LINE_RULE_SAMPLES];
  float aSemiCircleY[MAX_LINE_RULE_SAMPLES];
  float aSemiCircleW[MAX_LINE_RULE_SAMPLES];

  RadIntL state = {k};
  state.p.x = p.x;
  state.p.y = 0.0f;
  state.p.z = p.y;

  if (pOnElement) {
    assert(8 * 2 * nSections < MAX_LINE_RULE_SAMPLES);
    IntRule2D semiCircleRule = {8 * 2 * nSections, aSemiCircleX, aSemiCircleY, aSemiCircleW};
    semiCircleIntegralRule(2 * nSections, intRule, &semiCircleRule);
    state.semiCircleRule = semiCircleRule;

    if (k == 0.0f) {
      return complexQuadGenerator(integrateGeneratorL0_RAD, &state, intRule, p, a)
	+ complexQuadGenerator(integrateGeneratorL0_RAD, &state, intRule, p, b);
    } else {
      return computeL_RAD(0.0f, p, a, b, true)
	+ complexQuad2D(integrateGeneratorL0pOn_RAD, &state, intRule, &a, &b);
    }
  } else {
    assert(8 * nSections < MAX_LINE_RULE_SAMPLES);
    IntRule2D semiCircleRule = {8 * nSections, aSemiCircleX, aSemiCircleY, aSemiCircleW};
    semiCircleIntegralRule(nSections, intRule, &semiCircleRule);
    state.semiCircleRule = semiCircleRule;

    if (k == 0.0f) {
      return complexQuad2D(integrateGeneratorL0_RAD, &state, intRule, &a, &b);
    } else {
      return complexQuad2D(integrateGeneratorL_RAD, &state, intRule, &a, &b);
    }
  }
}

void ComputeL_RAD(float k, Float2 p, Float2 a, Float2 b, bool pOnElement, Complex * pResult) {
  std::complex<float> z = computeL_RAD(k, p, a, b, pOnElement);
  pResult->re = z.real();
  pResult->im = z.imag();
}

/* ---------------------------------------------------------------------------
 * Operator M
 */

std::complex<float> integrateSemiCircleM_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;
  Float3 q = {pS->r * x.x, pS->r * x.y, pS->z};
  Float3 nq = {pS->nq.x * x.x, pS->nq.x * x.y, pS->nq.y};
  Float3 r = q - pS->p;
  float R = norm(r);

  return (1if * pS->k * R - 1.0f) * std::exp(1if * pS->k * R) * dot(r, nq) / (R * dot(r, r));
}

std::complex<float> integrateGeneratorM_RAD(const Float2 *px, void *pState) {
  RadIntL * pS = (RadIntL *) pState;

  pS->r = px->x;
  pS->z = px->y;

  return complexLineIntegral(integrateSemiCircleM_RAD, pS, pS->semiCircleRule) * pS->r / static_cast<float>(2.0 * M_PI);
}

std::complex<float> integrateSemiCircleMpOn_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;
  Float3 q = {pS->r * x.x, pS->r * x.y, pS->z};
  Float3 nq = {pS->nq.x * x.x, pS->nq.x * x.y, pS->nq.y};
  Float3 r = q - pS->p;
  float R = norm(r);

  return -dot(r, nq) / (R * dot(r, r));
}

std::complex<float> integrateGeneratorMpOn_RAD(const Float2 *px, void *pState) {
  RadIntL * pS = (RadIntL *) pState;

  pS->r = px->x;
  pS->z = px->y;

  return complexLineIntegral(integrateSemiCircleMpOn_RAD, pS, pS->semiCircleRule) * pS->r / static_cast<float>(2.0 * M_PI);
}


/* Computes elements of the Helmholtz M-Operator radially symetrical cases.
 *
 * Parameters:
 *   k - the wavenumber of the problem.
 *   px, py - the point receiving radiation from the boundary.
 *   ar, az - the starting point of the boundary element being integrated over.
 *   br, bz - the end point of the bondary element being integrated over.
 *   pOnElement - a boolean indicating if p is on the boundary element being integrated.
 *
 *   Returns:
 *     The complex-valued result of the integration.
 */
std::complex<float> computeM_RAD(float k, Float2 p, Float2 a, Float2 b, bool pOnElement) {
  IntRule1D intRule = {8, aX_1D, aW_1D};

  /* subdivide circular integral into sections of similar size as qab */
  Float2 q = 0.5f * (a + b);
  Float2 ab = b - a;
  int nSections = 1 + (int)(q.x * M_PI / norm(ab));

  float aSemiCircleX[MAX_LINE_RULE_SAMPLES];
  float aSemiCircleY[MAX_LINE_RULE_SAMPLES];
  float aSemiCircleW[MAX_LINE_RULE_SAMPLES];

  assert(8 * nSections < MAX_LINE_RULE_SAMPLES);
  IntRule2D semiCircleRule = {8 * nSections, aSemiCircleX, aSemiCircleY, aSemiCircleW};
  semiCircleIntegralRule(nSections, intRule, &semiCircleRule);

  RadIntL state = {k};
  state.semiCircleRule = semiCircleRule;
  state.p.x = p.x;
  state.p.y = 0.0f;
  state.p.z = p.y;
  state.nq  = normal(a, b);

  if (k == 0.0f) {
    if (pOnElement) {
      return complexQuad2D(integrateGeneratorMpOn_RAD, &state, intRule, &a, &p)
	+ complexQuad2D(integrateGeneratorMpOn_RAD, &state, intRule, &p, &b);
    } else {
      return complexQuad2D(integrateGeneratorMpOn_RAD, &state, intRule, &a, &b);
    }
  } else {
    if (pOnElement) {
      return complexQuad2D(integrateGeneratorM_RAD, &state, intRule, &a, &p)
	+ complexQuad2D(integrateGeneratorM_RAD, &state, intRule, &p, &b);
    } else {
      return complexQuad2D(integrateGeneratorM_RAD, &state, intRule, &a, &b);
    }
  }
}


void ComputeM_RAD(float k, Float2 p, Float2 a, Float2 b, bool pOnElement, Complex * pResult) {
  std::complex<float> z = computeM_RAD(k, p, a, b, pOnElement);
  pResult->re = z.real();
  pResult->im = z.imag();
}

/* ---------------------------------------------------------------------------
 * Operator Mt
 */

std::complex<float> integrateSemiCircleMt_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;
  Float3 q = {pS->r * x.x, pS->r * x.y, pS->z};
  Float3 r = q - pS->p;
  float R = norm(r);
  float dotRnP = pS->np.x * r.x + pS->np.y * r.z;
  return -(1if * pS->k * R - 1.0f) * std::exp(1if * pS->k * R) * dotRnP / (R * dot(r, r));
}

std::complex<float> integrateGeneratorMt_RAD(const Float2 *px, void *pState) {
  RadIntL * pS = (RadIntL *) pState;

  pS->r = px->x;
  pS->z = px->y;

  return complexLineIntegral(integrateSemiCircleMt_RAD, pS, pS->semiCircleRule) * pS->r / static_cast<float>(2.0 * M_PI);
}

std::complex<float> integrateSemiCircleMtpOn_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;
  Float3 q = {pS->r * x.x, pS->r * x.y, pS->z};
  Float3 r = q - pS->p;
  float R = norm(r);
  float dotRnP = pS->np.x * r.x + pS->np.y * r.z;
  return dotRnP / (R * dot(r, r));
}

std::complex<float> integrateGeneratorMtpOn_RAD(const Float2 *px, void *pState) {
  RadIntL * pS = (RadIntL *) pState;

  pS->r = px->x;
  pS->z = px->y;

  return complexLineIntegral(integrateSemiCircleMtpOn_RAD, pS, pS->semiCircleRule) * pS->r / static_cast<float>(2.0 * M_PI);
}

/* Computes elements of the Helmholtz Mt-Operator radially symetrical cases.
 *
 * Parameters:
 *   k - the wavenumber of the problem.
 *   px, py - the point receiving radiation from the boundary.
 *   ar, az - the starting point of the boundary element being integrated over.
 *   br, bz - the end point of the bondary element being integrated over.
 *   pOnElement - a boolean indicating if p is on the boundary element being integrated.
 *
 *   Returns:
 *     The complex-valued result of the integration.
 */
std::complex<float> computeMt_RAD(float k, Float2 p, Float2 vec_p, Float2 a, Float2 b, bool pOnElement) {
  IntRule1D intRule = {8, aX_1D, aW_1D};

  /* subdivide circular integral into sections of similar size as qab */
  Float2 q = 0.5f * (a + b);
  Float2 ab = b - a;
  int nSections = 1 + (int)(q.x * M_PI / norm(ab));

  float aSemiCircleX[MAX_LINE_RULE_SAMPLES];
  float aSemiCircleY[MAX_LINE_RULE_SAMPLES];
  float aSemiCircleW[MAX_LINE_RULE_SAMPLES];

  assert(8 * nSections < MAX_LINE_RULE_SAMPLES);
  IntRule2D semiCircleRule = {8 * nSections, aSemiCircleX, aSemiCircleY, aSemiCircleW};
  semiCircleIntegralRule(nSections, intRule, &semiCircleRule);

  RadIntL state = {k};
  state.semiCircleRule = semiCircleRule;
  state.p.x = p.x;
  state.p.y = 0.0f;
  state.p.z = p.y;
  state.nq  = normal(a, b);
  state.np  = vec_p;

  if (k == 0.0f) {
    if (pOnElement) {
      return complexQuad2D(integrateGeneratorMtpOn_RAD, &state, intRule, &a, &p)
	+ complexQuad2D(integrateGeneratorMtpOn_RAD, &state, intRule, &p, &b);
    } else {
      return complexQuad2D(integrateGeneratorMtpOn_RAD, &state, intRule, &a, &b);
    }
  } else {
    if (pOnElement) {
      return complexQuad2D(integrateGeneratorMt_RAD, &state, intRule, &a, &p)
	+ complexQuad2D(integrateGeneratorMt_RAD, &state, intRule, &p, &b);
    } else {
      return complexQuad2D(integrateGeneratorMt_RAD, &state, intRule, &a, &b);
    }
  }
}


void ComputeMt_RAD(float k, Float2 p, Float2 vec_p, Float2 a, Float2 b, bool pOnElement, Complex * pResult) {
  std::complex<float> z = computeMt_RAD(k, p, vec_p, a, b, pOnElement);
  pResult->re = z.real();
  pResult->im = z.imag();
}

/* ---------------------------------------------------------------------------
 * Operator N
 */

std::complex<float> integrateSemiCircleN_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;
  Float3 q  = {pS->r * x.x, pS->r * x.y, pS->z};
  Float3 nq = {pS->nq.x * x.x, pS->nq.x * x.y, pS->nq.y};
  Float3 r = q - pS->p;
  float R = norm(r);
  float dotnPnQ = pS->np.x * nq.x + pS->np.y * nq.z;
  float dotRnP  = pS->np.x * r.x + pS->np.y * r.z;
  float dotRnQ  = -dot(r, nq);
  float RnPRnQ  = dotRnP * dotRnQ / dot(r, r);
  float RnPnQ   = -(dotnPnQ + RnPRnQ) / R;
  std::complex<float> ikr = 1if * pS->k * R;
  std::complex<float> fpgr = std::exp(ikr) / dot(r, r) * (ikr - 1.0f);
  std::complex<float> fpgrr = std::exp(ikr) * (2.0f - 2.0f * ikr - (pS->k * R)*(pS->k * R)) / (dot(r, r) * R);
  return fpgr * RnPnQ + fpgrr * RnPRnQ;
}

std::complex<float> integrateGeneratorN_RAD(const Float2 *px, void *pState) {
  RadIntL * pS = (RadIntL *) pState;

  pS->r = px->x;
  pS->z = px->y;

  return complexLineIntegral(integrateSemiCircleN_RAD, pS, pS->semiCircleRule) * pS->r / static_cast<float>(2.0 * M_PI);
}

std::complex<float> integrateSemiCircleNpOn_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;
  Float3 q  = {pS->r * x.x, pS->r * x.y, pS->z};
  Float3 nq = {pS->nq.x * x.x, pS->nq.x * x.y, pS->nq.y};
  Float3 r  = q - pS->p;
  float  R  = norm(r);
  float dotnPnQ = pS->np.x * nq.x + pS->np.y * nq.z;
  float dotRnP  = pS->np.x * r.x + pS->np.y * r.z;
  float dotRnQ  = -dot(r, nq);
  float RnPRnQ  = dotRnP * dotRnQ / dot(r, r);
  float RnPnQ   = -(dotnPnQ + RnPRnQ) / R;
  std::complex<float> ikr    = 1if * pS->k * R;
  float         fpg0   = 1.0f / R;
  std::complex<float> fpgr   = std::exp(ikr) / dot(r, r) * (ikr - 1.0f);
  float         fpgr0  = -1.0f / dot(r, r);
  std::complex<float> fpgrr  = std::exp(ikr) * (2.0f - 2.0f * ikr - (pS->k * R)*(pS->k * R)) / (dot(r, r) * R);
  float         fpgrr0 = 2.0f / (R * dot(r, r));
  return (fpgr-fpgr0) * RnPnQ + (fpgrr-fpgrr0) * RnPRnQ + 0.5f * (pS->k*pS->k) * fpg0;
}

std::complex<float> integrateGeneratorNpOn_RAD(const Float2 *px, void *pState) {
  RadIntL * pS = (RadIntL *) pState;

  pS->r = px->x;
  pS->z = px->y;

  return complexLineIntegral(integrateSemiCircleNpOn_RAD, pS, pS->semiCircleRule) * pS->r / static_cast<float>(2.0 * M_PI);
}

std::complex<float> integrateSemiCircleN0pOn_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;
  Float3 q  = {pS->r * x.x, pS->r * x.y, pS->z};
  Float3 nq = {x.x, x.y, static_cast<float>(pS->direction)};
  nq = sqrtf(0.5f) * nq;
  Float3 r  = q - pS->p;
  float  R  = norm(r);
  float dotnPnQ = pS->np.x * nq.x + pS->np.y * nq.z;
  float dotRnP  = pS->np.x * r.x + pS->np.y * r.z;
  float dotRnQ  = -dot(r, nq);
  float RnPRnQ  = dotRnP * dotRnQ / dot(r, r);
  return (dotnPnQ + 3.0f * RnPRnQ) / (R * dot(r,r));
}

std::complex<float> integrateGeneratorN0pOn_RAD(const Float2 *px, void *pState) {
  RadIntL * pS = (RadIntL *) pState;

  pS->r = px->x;
  pS->z = px->y;

  return complexLineIntegral(integrateSemiCircleN0pOn_RAD, pS, pS->semiCircleRule) * pS->r / static_cast<float>(2.0 * M_PI);
}

std::complex<float> complexConeIntegral(std::complex<float>(*integrand)(const Float2*, void*), void* state, IntRule1D intRule,
				  Float2 start, Float2 end, int nSections) {
  Float2 delta = 1.0f/nSections * (end - start);
  std::complex<float> sum = 0.0f;
  for (int s = 0; s < nSections; ++s) {
    Float2 segmentStart = start + static_cast<float>(s) * delta;
    Float2 segmentEnd   = start + static_cast<float>(s+1) * delta;
    sum += complexQuad2D(integrand, state, intRule, &segmentStart, &segmentEnd);
  }
  return sum;
}

/* Computes elements of the Helmholtz N-Operator radially symetrical cases.
 *
 * Parameters:
 *   k - the wavenumber of the problem.
 *   px, py - the point receiving radiation from the boundary.
 *   ar, az - the starting point of the boundary element being integrated over.
 *   br, bz - the end point of the bondary element being integrated over.
 *   pOnElement - a boolean indicating if p is on the boundary element being integrated.
 *
 *   Returns:
 *     The complex-valued result of the integration.
 */
std::complex<float> computeN_RAD(float k, Float2 p, Float2 vec_p, Float2 a, Float2 b, bool pOnElement) {
  IntRule1D intRule = {8, aX_1D, aW_1D};

  /* subdivide circular integral into sections of similar size as qab */
  Float2 q = 0.5f * (a + b);
  Float2 ab = b - a;
  int nSections = 1 + (int)(q.x * M_PI / norm(ab));

  float aSemiCircleX[MAX_LINE_RULE_SAMPLES];
  float aSemiCircleY[MAX_LINE_RULE_SAMPLES];
  float aSemiCircleW[MAX_LINE_RULE_SAMPLES];

  assert(8 * nSections < MAX_LINE_RULE_SAMPLES);
  IntRule2D semiCircleRule = {8 * nSections, aSemiCircleX, aSemiCircleY, aSemiCircleW};
  semiCircleIntegralRule(nSections, intRule, &semiCircleRule);

  RadIntL state = {k};
  state.semiCircleRule = semiCircleRule;
  state.p.x = p.x;
  state.p.y = 0.0f;
  state.p.z = p.y;
  state.nq  = normal(a, b);
  state.np  = vec_p;

  if (k == 0.0f) {
    if (pOnElement) {
      float lenAB = norm(b - a);
      /* deal with the cone at the a-side of the generator */
      int direction = -1;
      if (a.y >= b.y) direction = 1;
      Float2 tip_a = {0.0f, a.y + direction * a.x};
      state.direction = direction;
      int nSections = (int)(a.x * sqrtf(2.0f) / lenAB) + 1;
      std::complex<float> coneValA = complexConeIntegral(integrateGeneratorN0pOn_RAD, &state, intRule, a, tip_a, nSections);

      /* deal with the cone at the b-side of the generator */
      Float2 tip_b = {0.0, b.y - direction * b.x};
      state.direction = -direction;
      nSections = static_cast<int>(b.x * sqrtf(2.0f) / lenAB) + 1;
      std::complex<float> coneValB = complexConeIntegral(integrateGeneratorN0pOn_RAD, &state, intRule, b, tip_b, nSections);

      return -(coneValA + coneValB);
    } else {
      return 0.0f;
    }
  } else {
    if (pOnElement) {
      return computeN_RAD(0.0f, p, vec_p, a, b, true) - 0.5f * (k*k) * computeL_RAD(0.0f, p, a, b, true)
	+ complexQuad2D(integrateGeneratorNpOn_RAD, &state, intRule, &a, &p)
	+ complexQuad2D(integrateGeneratorNpOn_RAD, &state, intRule, &p, &b);
    } else {
      return complexQuad2D(integrateGeneratorN_RAD, &state, intRule, &a, &b);
    }
  }
}

void ComputeN_RAD(float k, Float2 p, Float2 vec_p, Float2 a, Float2 b, bool pOnElement, Complex * pResult) {
  std::complex<float> z = computeN_RAD(k, p, vec_p, a, b, pOnElement);
  pResult->re = z.real();
  pResult->im = z.imag();
}


/* --------------------------------------------------------------------------  */
/*                  3D discrete Helmholtz operators.                           */
/* --------------------------------------------------------------------------- */

typedef struct {
  float k;
  Float3 p;
  Float3 normal_p;
  Float3 normal_q;
} IntL3D;

std::complex<float> intL1_3D(Float3 x, void* state) {
  IntL3D * s = (IntL3D *) state;
  float R = norm(s->p - x);
  return std::exp(1if * s->k * R) / R;
}

std::complex<float> intL2_3D(Float3 x, void* state) {
  IntL3D * s = (IntL3D *) state;
  float R = norm(s->p - x);
  return 1.0f / R;
}

std::complex<float> intL3_3D(Float3 x, void* state) {
  IntL3D * s = (IntL3D *) state;
  float R = norm(s->p - x);
  return (std::exp(1if * s->k * R) - 1.0f) / R;
}

 /* Computes elements of the Helmholtz L-Operator for 3D.
 *
 * Parameters:
 *   k - the wavenumber of the problem.
 *   px, py, pz - the point receiving radiation from the boundary.
 *   ax, ay, az - the first vertex of ccw triangle.
 *   bx, by, bz - the second vertex of ccw triangle.
 *   cx, cy, cz - the third vertex of ccw t riangle.
 *   pOnElement - a boolean indicating if p is on the boundary element being integrated.
 *
 *   Returns:
 *     The results of the integration, which are typically a complex
 *     number are stored in those two floats, the first being the Real-, the
 *     second being the Imaginary component of that complex value.
 */
std::complex<float> computeL_3D(float k, Float3 p, Float3 a, Float3 b, Float3 c, bool pOnElement) {
  IntL3D stat = {k, p};
  IntRule2D intRule = {7, aX_2D, aY_2D, aW_2D};

  if (pOnElement) {
    if (k == 0.0f) {
      Float3 ab = b - a;
      Float3 ac = c - a;
      Float3 bc = c - b;

      float aopp[3] = {norm(ab), norm(bc), norm(ac)};

      Float3 ap = p - a;
      Float3 bp = p - b;
      Float3 cp = p - c;

      float ar0[3] = {norm(ap), norm(bp), norm(cp)};
      float ara[3] = {ar0[1], ar0[2], ar0[0]};

      float result = 0.0f;
      for (int i = 0; i < 3; ++i) {
	float r0 = ar0[i];
	float ra = ara[i];
	float opp = aopp[i];
	if (r0 < ra) {
	  float temp = r0;
	  r0 = ra;
	  ra = temp;
	}
	float A = acosf((ra*ra + r0*r0 - opp*opp) / (2.0f * ra * r0));
	float B = atanf(ra * sinf(A) / (r0 - ra * cosf(A)));
	result += (r0 * sinf(B) * (logf(tanf(0.5f * (A + B))) - logf(tanf(0.5f * B))));
      }
      return result / (4.0f * M_PI);
    } else {
      std::complex<float> L0 = computeL_3D(0.0, p, a, b, c, true);
      std::complex<float> Lk = complexQuad3D(intL3_3D, &stat, intRule, a, b, p)
	+ complexQuad3D(intL3_3D, &stat, intRule, b, c, p)
	+ complexQuad3D(intL3_3D, &stat, intRule, c, a, p);
      return L0 + Lk / static_cast<float>(4.0 * M_PI);

    }
  } else {
    if (k == 0.0f) {
      return complexQuad3D(intL2_3D, &stat, intRule, a, b, c) / static_cast<float>(4.0 * M_PI);
    } else {
      return complexQuad3D(intL1_3D, &stat, intRule, a, b, c) / static_cast<float>(4.0 * M_PI);
    }
  }
}

void ComputeL_3D(float k, Float3 p, Float3 a, Float3 b, Float3 c, bool pOnElement, Complex * pResult) {
  std::complex<float> z = computeL_3D(k, p, a, b, c, pOnElement);
  pResult->re = z.real();
  pResult->im = z.imag();
}

std::complex<float> intM1_3D(Float3 x, void* state) {
  IntL3D * s = (IntL3D *) state;
  Float3 r = s->p - x;
  float R = norm(r);
  float kr = s->k * R;
  std::complex<float> ikr = 1if * kr;
  float rnq = -dot(r, s->normal_q) / R;
  return rnq * (ikr - 1.0f) * std::exp(ikr) / dot(r, r);
}

std::complex<float> intM2_3D(Float3 x, void* state) {
  IntL3D * s = (IntL3D *) state;
  Float3 r = s->p - x;
  float R = norm(r);
  float rnq = -dot(r, s->normal_q) / R;
  return -1.0f / dot(r, r) * rnq;
}

 /* Computes elements of the Helmholtz M-Operator.
 *
 * Parameters:
 *   k - the wavenumber of the problem.
 *   px, py, pz - the point receiving radiation from the boundary.
 *   a, b, c - the three vertices forming the triangle being integrated over.
 *   pOnElement - a boolean indicating if p is on the boundary element being integrated.
 *
 *   Returns:
 *     The results of the integration, which are typically a complex
 *     number are stored in those two floats, the first being the Real-, the
 *     second being the Imaginary component of that complex value.
 */
std::complex<float> computeM_3D(float k, Float3 p, Float3 a, Float3 b, Float3 c, bool pOnElement) {
  IntL3D stat = {k, p};
  stat.normal_q = Normal3D(a, b, c);
  IntRule2D intRule = {7, aX_2D, aY_2D, aW_2D};

  if (pOnElement) {
    return 0.0f;
  } else {
    if (k == 0.0f) {
      return complexQuad3D(intM2_3D, &stat, intRule, a, b, c) / static_cast<float>(4.0 * M_PI);
    } else {
      return complexQuad3D(intM1_3D, &stat, intRule, a, b, c) / static_cast<float>(4.0 * M_PI);
    }
  }
}

void ComputeM_3D(float k, Float3 p, Float3 a, Float3 b, Float3 c, bool pOnElement, Complex * pResult) {
  std::complex<float> z = computeM_3D(k, p, a, b, c, pOnElement);
  pResult->re = z.real();
  pResult->im = z.imag();
}


std::complex<float> intMt1_3D(Float3 x, void* state) {
  IntL3D * s = (IntL3D *) state;
  Float3 r = s->p - x;
  float R = norm(r);
  float kr = s->k * R;
  std::complex<float> ikr = 1if * kr;
  float rnp = dot(r, s->normal_p) / R;
  return rnp * (ikr - 1.0f) * std::exp(ikr) / dot(r, r);
}

std::complex<float> intMt2_3D(Float3 x, void* state) {
  IntL3D * s = (IntL3D *) state;
  Float3 r = s->p - x;
  float R = norm(r);
  float rnp = dot(r, s->normal_p) / R;
  return -1.0f / dot(r, r) * rnp;
}

 /* Computes elements of the Helmholtz Mt-Operator.
 *
 * Parameters:
 *   k - the wavenumber of the problem.
 *   px, py, pz - the point receiving radiation from the boundary.
 *   vec_p - the surface normal in p.
 *   a, b, c - the three vertices forming the triangle being integrated over.
 *   pOnElement - a boolean indicating if p is on the boundary element being integrated.
 *
 *   Returns:
 *     The results of the integration, which are typically a complex
 *     number are stored in those two floats, the first being the Real-, the
 *     second being the Imaginary component of that complex value.
 */
std::complex<float> computeMt_3D(float k, Float3 p, Float3 vec_p, Float3 a, Float3 b, Float3 c, bool pOnElement) {
  IntL3D stat = {k, p};
  stat.normal_p = vec_p;
  IntRule2D intRule = {7, aX_2D, aY_2D, aW_2D};

  if (pOnElement) {
    return 0.0f;
  } else {
    if (k == 0.0f) {
      return complexQuad3D(intMt2_3D, &stat, intRule, a, b, c) / static_cast<float>(4.0 * M_PI);
    } else {
      return complexQuad3D(intMt1_3D, &stat, intRule, a, b, c) / static_cast<float>(4.0 * M_PI);
    }
  }
}

void ComputeMt_3D(float k, Float3 p, Float3 vec_p, Float3 a, Float3 b, Float3 c, bool pOnElement, Complex * pResult) {
  std::complex<float> z = computeMt_3D(k, p, vec_p, a, b, c, pOnElement);
  pResult->re = z.real();
  pResult->im = z.imag();
}


std::complex<float> intN1_3D(Float3 x, void* state) {
  IntL3D * s = (IntL3D *) state;
  Float3 r = s->p - x;
  float R  = norm(r);
  float kr = s->k * R;
  std::complex<float> ikr = 1if * kr;

  float rnq    = -dot(r, s->normal_q) / R;
  float rnp    =  dot(r, s->normal_p) / R;
  float dnpnq  =  dot(s->normal_p, s->normal_q);
  float rnprnq = rnp * rnq;
  float rnpnq  = -(dnpnq + rnprnq) / R;

  std::complex<float> fpgr  = (ikr - 1.0f) * std::exp(ikr) / dot(r, r);
  std::complex<float> fpgrr = std::exp(ikr) * (2.0f - 2.0f*ikr - kr*kr) / (R * dot(r, r));
  return fpgr * rnpnq + fpgrr * rnprnq;
}

std::complex<float> intN2_3D(Float3 x, void* state) {
  IntL3D * s = (IntL3D *) state;
  Float3 r = s->p - x;
  float R  = norm(r);
  float kr = s->k * R;
  std::complex<float> ikr = 1if * kr;

  float rnq    = -dot(r, s->normal_q) / R;
  float rnp    =  dot(r, s->normal_p) / R;
  float dnpnq  =  dot(s->normal_p, s->normal_q);
  float rnprnq = rnp * rnq;
  float rnpnq  = -(dnpnq + rnprnq) / R;


          float fpg   = 1.0f / R;
  std::complex<float> fpgr  = ((ikr - 1.0f) * std::exp(ikr) + 1.0f) / dot(r, r);
  std::complex<float> fpgrr = (std::exp(ikr) * (2.0f - 2.0f*ikr - kr*kr) - 2.0f) / (R * dot(r, r));
  return fpgr * rnpnq + fpgrr * rnprnq + (0.5f*s->k*s->k) * fpg;
}

/* Computes elements of the Helmholtz N-Operator.
 *
 * Parameters:
 *   k - the wavenumber of the problem.
 *   px, py, pz - the point receiving radiation from the boundary.
 *   vec_p - the surface normal in point p.
 *   a, b, c - the vertices of the trinagle being integrated over.
 *   pOnElement - a boolean indicating if p is on the boundary element being integrated.
 *
 *   Returns:
 *     The results of the integration, which are typically a complex
 *     number are stored in those two floats, the first being the Real-, the
 *     second being the Imaginary component of that complex value.
 */
std::complex<float> computeN_3D(float k, Float3 p, Float3 vec_p, Float3 a, Float3 b, Float3 c, bool pOnElement) {
  IntL3D stat = {k, p};
  stat.normal_p = vec_p;
  stat.normal_q = Normal3D(a, b, c);
  IntRule2D intRule = {7, aX_2D, aY_2D, aW_2D};

  if (pOnElement) {
    if (k == 0.0f) {
      Float3 ab = b - a;
      Float3 ac = c - a;
      Float3 bc = c - b;

      float aopp[3] = {norm(ab), norm(bc), norm(ac)};

      Float3 ap = p - a;
      Float3 bp = p - b;
      Float3 cp = p - c;

      float ar0[3] = {norm(ap), norm(bp), norm(cp)};
      float ara[3] = {ar0[1], ar0[2], ar0[0]};

      float result = 0.0f;
      for (int i = 0; i < 3; ++i) {
	float r0 = ar0[i];
	float ra = ara[i];
	float opp = aopp[i];
	if (r0 < ra) {
	  float temp = r0;
	  r0 = ra;
	  ra = temp;
	}
	float A = acosf((ra*ra + r0*r0 - opp*opp) / (2.0f * ra * r0));
	float B = atanf(ra * sinf(A) / (r0 - ra * cosf(A)));
	result += (cosf(A + B) - cosf(B)) / (r0 * sinf(B));
      }
      return result / (4.0f * M_PI);
    } else {
      std::complex<float> N0 = computeN_3D(0.0f, p, vec_p, a, b, c, true);
      std::complex<float> L0 = computeL_3D(0.0f, p, a, b, c, true);
      std::complex<float> Nk = complexQuad3D(intN2_3D, &stat, intRule, a, b, p)
	+ complexQuad3D(intN2_3D, &stat, intRule, b, c, p)
	+ complexQuad3D(intN2_3D, &stat, intRule, c, a, p);
	return N0 - (0.5f*k*k) * L0 + Nk / static_cast<float>(4.0 * M_PI);
    }
  } else {
    if (k == 0.0f) {
      return 0.0f;
    } else {
      return complexQuad3D(intN1_3D, &stat, intRule, a, b, c) / static_cast<float>(4.0 * M_PI);
    }
  }
}

void ComputeN_3D(float k, Float3 p, Float3 normal_p, Float3 a, Float3 b, Float3 c, bool pOnElement, Complex * pResult) {
  std::complex<float> z = computeN_3D(k, p, normal_p, a, b, c, pOnElement);
  pResult->re = z.real();
  pResult->im = z.imag();
}
