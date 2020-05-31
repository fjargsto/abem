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

std::complex<float> complexQuad2D(std::complex<float> (*integrand)(Float2, void*), void * state, IntRule1D intRule,
			    Float2 start, Float2 end) {
  Float2 vec;
  vec = end - start;
  std::complex<float> sum = 0.0f;
  for (int i = 0; i < intRule.nSamples; ++i)
    sum += intRule.pW[i] * integrand(intRule.pX[i] * vec + start, state);

  return norm(vec) * sum;
}

std::complex<float> complexQuadGenerator(std::complex<float> (*integrand)(Float2, void*), void * state, IntRule1D intRule,
				   Float2 start, Float2 end) {
  Float2 vec;
  vec = end - start;
  std::complex<float> sum = 0.0f;
  for (int i = 0; i < intRule.nSamples; ++i) {
    float x_sqr = intRule.pX[i] * intRule.pX[i];
    sum += intRule.pW[i] * integrand(x_sqr * vec + start, state) * intRule.pX[i];
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

/* --------------------------------------------------------------------------  */
/*                       2D discrete Helmholtz operators.                      */
/* --------------------------------------------------------------------------- */

typedef struct {
  float k;
  Float2 p;
  Float2 normal_p;
  Float2 normal_q;
} IntL;

std::complex<float> intL1_2D(Float2 x, void* state) {
  IntL * s = (IntL *) state;
  float R = norm(s->p - x);
  return static_cast<float>(0.5f * M_1_PI * logf(R)) + 0.25f * 1if * hankel1(0, s->k * R);
}

std::complex<float> intL2_2D(Float2 x, void* state) {
  IntL * s = (IntL *) state;
  float R = norm(s->p - x);
  return logf(R);
}

std::complex<float> intL3_2D(Float2 x, void* state) {
  IntL * s = (IntL *) state;
  float R = norm(s->p - x);
  return hankel1(0, s->k * R);
}

/* Computes elements of the Helmholtz L-Operator for 2D.
 *
 * Parameters:
 *   k - the wavenumber of the problem.
 *   px, py - the point receiving radiation from the boundary.
 *   ax, ay - the starting point of the boundary element being integrated over.
 *   bx, by - the end point of the bondary element being integrated over.
 *   pOnElement - a boolean indicating if p is on the boundary element being integrated.
 *
 *   Returns:
 *     The results of the integration, which are typically a complex
 *     number are stored in those two floats, the first being the Real-, the
 *     second being the Imaginary component of that complex value.
 */
std::complex<float> computeL_2D(float k, Float2 p, Float2 a, Float2 b, bool pOnElement) {
  IntL stat = {k, p};
  IntRule1D intRule = {8, aX_1D, aW_1D};

  Float2 ab = b - a;
  if (pOnElement) {
    if (k == 0.0f) {
      float RA = norm(p - a);
      float RB = norm(p - b);
      float RAB = norm(ab);
      return 0.5f * M_1_PI * (RAB - (RA * logf(RA) + RB * logf(RB)));
    } else {
      return complexQuad2D(intL1_2D, &stat, intRule, a, p) + complexQuad2D(intL1_2D, &stat, intRule, p, b)
	+ computeL_2D(0, p, a, b, true);
    }
  } else {
    if (k == 0.0f) {
      return static_cast<float>(-0.5f * M_1_PI) * complexQuad2D(intL2_2D, &stat, intRule, a, b);
    } else {
      return 0.25f * 1if * complexQuad2D(intL3_2D, &stat, intRule, a, b);
    }
  }
}

void ComputeL_2D(float k, Float2 p, Float2 a, Float2 b, bool pOnElement, Complex * pResult) {
  std::complex<float> z = computeL_2D(k, p, a, b, pOnElement);
  pResult->re = z.real();
  pResult->im = z.imag();
}


std::complex<float> intM1_2D(Float2 x, void* state) {
  IntL * s = (IntL *) state;
  Float2 r = s->p - x;
  return dot(r, s->normal_q) / dot(r, r);
}

std::complex<float> intM2_2D(Float2 x, void* state) {
  IntL * s = (IntL *) state;
  Float2 r = s->p - x;
  float R = norm(r);
  return hankel1(1, s->k * R) * dot(r, s->normal_q) / R;
}

/* Computes elements of the Helmholtz M-Operator.
 *
 * Parameters:
 *   k - the wavenumber of the problem.
 *   px, py - the point receiving radiation from the boundary.
 *   ax, ay - the starting point of the boundary element being integrated over.
 *   bx, by - the end point of the bondary element being integrated over.
 *   pOnElement - a boolean indicating if p is on the boundary element being integrated.
 *
 *   Returns:
 *     The results of the integration, which are typically a complex
 *     number are stored in those two floats, the first being the Real-, the
 *     second being the Imaginary component of that complex value.
 */
std::complex<float> computeM_2D(float k, Float2 p, Float2 a, Float2 b, bool pOnElement) {
  Float2 zero = {0.0f, 0.0f};
  IntL stat = {k, p, zero, normal(a, b)};
  IntRule1D intRule = {8, aX_1D, aW_1D};

  if (pOnElement) {
    return 0.0;
  } else {
    if (k == 0.0f) {
      return static_cast<float>(-0.5f * M_1_PI) * complexQuad2D(intM1_2D, &stat, intRule, a, b);
    } else {
      return 0.25f * 1if * k * complexQuad2D(intM2_2D, &stat, intRule, a, b);
    }
  }
}

void ComputeM_2D(float k, Float2 p, Float2 a, Float2 b, bool pOnElement, Complex * pResult) {
  std::complex<float> z = computeM_2D(k, p, a, b, pOnElement);
  pResult->re = z.real();
  pResult->im = z.imag();
}

/* Computes elements of the Helmholtz Mt-Operator.
 *
 * Parameters:
 *   k - the wavenumber of the problem.
 *   px, py - the point receiving radiation from the boundary.
 *   ax, ay - the starting point of the boundary element being integrated over.
 *   bx, by - the end point of the bondary element being integrated over.
 *   pOnElement - a boolean indicating if p is on the boundary element being integrated.
 *
 *   Returns:
 *     The results of the integration, which are typically a complex
 *     number are stored in those two floats, the first being the Real-, the
 *     second being the Imaginary component of that complex value.
 */
std::complex<float> computeMt_2D(float k, Float2 p, Float2 normal_p, Float2 a, Float2 b, bool pOnElement) {
  /* The flollowing is a little hacky, as we're not storing the actual normal_p vector in the
   * normal_q field of the state struct. By doing this we can reuse the two functions for the
   * M operator's integral evaluation intM1 and intM2.
   */
  Float2 zero = {0.0f, 0.0f};
  IntL stat = {k, p, zero, normal_p};
  IntRule1D intRule = {8, aX_1D, aW_1D};

  if (pOnElement) {
    return 0.0;
  } else {
    if (k == 0.0f) {
      return static_cast<float>(-0.5f * M_1_PI) * complexQuad2D(intM1_2D, &stat, intRule, a, b);
    } else {
      return -0.25f * 1if * k * complexQuad2D(intM2_2D, &stat, intRule, a, b);
    }
  }
}

void ComputeMt_2D(float k, Float2 p, Float2 normal_p, Float2 a, Float2 b, bool pOnElement, Complex * pResult) {
  std::complex<float> z = computeMt_2D(k, p, normal_p, a, b, pOnElement);
  pResult->re = z.real();
  pResult->im = z.imag();
}


std::complex<float> intN1_2D(Float2 x, void* state) {
  IntL * s = (IntL *) state;
  Float2 r = s->p - x;
  float R2 = dot(r, r);
  float R = sqrtf(R2);
  float drdudrdn = -dot(r, s->normal_q) * dot(r, s->normal_p) / R2;
  float dpnu = dot(s->normal_p, s->normal_q);
  std::complex<float> c1 = 0.25f * 1if * s->k / R * hankel1(1, s->k * R) - 0.5f * static_cast<float>(M_1_PI / R2);
  std::complex<float> c2 = 0.50f * 1if * s->k / R * hankel1(1, s->k * R)
    - 0.25f * 1if * s->k * s->k * hankel1(0, s->k * R) - static_cast<float>(M_1_PI / R2);
  float c3 = -0.25f * s->k * s->k * logf(R) * static_cast<float>(M_1_PI);

  return c1 * dpnu + c2 * drdudrdn + c3;
}

std::complex<float> intN2_2D(Float2 x, void* state) {
  IntL * s = (IntL *) state;
  Float2 r = s->p - x;
  float R2 = dot(r, r);
  float drdudrdn = -dot(r, s->normal_q) * dot(r, s->normal_p) / R2;
  float dpnu = dot(s->normal_p, s->normal_q);

  return (dpnu + 2.0f * drdudrdn) / R2;
}

std::complex<float> intN3_2D(Float2 x, void* state) {
  IntL * s = (IntL *) state;
  Float2 r = s->p - x;
  float R2 = dot(r, r);
  float R = sqrtf(R2);
  float drdudrdn = -dot(r, s->normal_q) * dot(r, s->normal_p) / R2;
  float dpnu = dot(s->normal_p, s->normal_q);

  return hankel1(1, s->k * R) / R * (dpnu + 2.0f * drdudrdn)
    - s->k * hankel1(0, s->k * R) * drdudrdn;
}


/* Computes elements of the Helmholtz N-Operator.
 *
 * Parameters:
 *   k - the wavenumber of the problem.
 *   px, py - the point receiving radiation from the boundary.
 *   ax, ay - the starting point of the boundary element being integrated over.
 *   bx, by - the end point of the bondary element being integrated over.
 *   pOnElement - a boolean indicating if p is on the boundary element being integrated.
 *
 *   Returns:
 *     The results of the integration, which are typically a complex
 *     number are stored in those two floats, the first being the Real-, the
 *     second being the Imaginary component of that complex value.
 */
std::complex<float> computeN_2D(float k, Float2 p, Float2 normal_p, Float2 a, Float2 b, bool pOnElement) {
  IntL stat = {k, p, normal_p, normal(a, b)};
  IntRule1D intRule = {8, aX_1D, aW_1D};

  if (pOnElement) {
    if (k == 0.0f) {
      float RA = norm(p - a);
      float RB = norm(p - b);
      float RAB = norm(b - a);
      return -(1.0f / RA + 1.0f / RB) / (RAB * 2.0 * M_PI) * RAB;
    } else {
      return computeN_2D(0.0f, p, normal_p, a, b, true)
	- 0.5f * k * k * computeL_2D(0.0f, p, a, b, true)
	+ complexQuad2D(intN1_2D, &stat, intRule, a, p) + complexQuad2D(intN1_2D, &stat, intRule, p, b);
    }
  } else {
    if (k == 0.0f) {
      return static_cast<float>(0.5 * M_1_PI) * complexQuad2D(intN2_2D, &stat, intRule, a, b);
    } else {
      return 0.25f * 1if * k * complexQuad2D(intN3_2D, &stat, intRule, a, b);
    }
  }
}

void ComputeN_2D(float k, Float2 p, Float2 normal_p, Float2 a, Float2 b, bool pOnElement, Complex * pResult) {
  std::complex<float> z = computeN_2D(k, p, normal_p, a, b, pOnElement);
  pResult->re = z.real();
  pResult->im = z.imag();
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

std::complex<float> integrateGeneratorL_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;

  pS->r = x.x;
  pS->z = x.y;

  return complexLineIntegral(integrateSemiCircleL_RAD, pS, pS->semiCircleRule) * pS->r / static_cast<float>(2.0 * M_PI);
}

std::complex<float> integrateSemiCircleL0_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;
  Float3 q = {pS->r * x.x, pS->r * x.y, pS->z};
  float R = norm(q - pS->p);

  return 1.0f / R;
}

std::complex<float> integrateGeneratorL0_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;

  pS->r = x.x;
  pS->z = x.y;

  return complexLineIntegral(integrateSemiCircleL0_RAD, pS, pS->semiCircleRule) * pS->r / static_cast<float>(2.0f * M_PI);
}

std::complex<float> integrateSemiCircleL0pOn_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;
  Float3 q = {pS->r * x.x, pS->r * x.y, pS->z};
  float R = norm(q - pS->p);

  return (std::exp(1if * pS->k * R) - 1.0f) / R;
}

std::complex<float> integrateGeneratorL0pOn_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;

  pS->r = x.x;
  pS->z = x.y;

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
	+ complexQuad2D(integrateGeneratorL0pOn_RAD, &state, intRule, a, b);
    }
  } else {
    assert(8 * nSections < MAX_LINE_RULE_SAMPLES);
    IntRule2D semiCircleRule = {8 * nSections, aSemiCircleX, aSemiCircleY, aSemiCircleW};
    semiCircleIntegralRule(nSections, intRule, &semiCircleRule);
    state.semiCircleRule = semiCircleRule;

    if (k == 0.0f) {
      return complexQuad2D(integrateGeneratorL0_RAD, &state, intRule, a, b);
    } else {
      return complexQuad2D(integrateGeneratorL_RAD, &state, intRule, a, b);
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

std::complex<float> integrateGeneratorM_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;

  pS->r = x.x;
  pS->z = x.y;

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

std::complex<float> integrateGeneratorMpOn_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;

  pS->r = x.x;
  pS->z = x.y;

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
      return complexQuad2D(integrateGeneratorMpOn_RAD, &state, intRule, a, p)
	+ complexQuad2D(integrateGeneratorMpOn_RAD, &state, intRule, p, b);
    } else {
      return complexQuad2D(integrateGeneratorMpOn_RAD, &state, intRule, a, b);
    }
  } else {
    if (pOnElement) {
      return complexQuad2D(integrateGeneratorM_RAD, &state, intRule, a, p)
	+ complexQuad2D(integrateGeneratorM_RAD, &state, intRule, p, b);
    } else {
      return complexQuad2D(integrateGeneratorM_RAD, &state, intRule, a, b);
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

std::complex<float> integrateGeneratorMt_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;

  pS->r = x.x;
  pS->z = x.y;

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

std::complex<float> integrateGeneratorMtpOn_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;

  pS->r = x.x;
  pS->z = x.y;

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
      return complexQuad2D(integrateGeneratorMtpOn_RAD, &state, intRule, a, p)
	+ complexQuad2D(integrateGeneratorMtpOn_RAD, &state, intRule, p, b);
    } else {
      return complexQuad2D(integrateGeneratorMtpOn_RAD, &state, intRule, a, b);
    }
  } else {
    if (pOnElement) {
      return complexQuad2D(integrateGeneratorMt_RAD, &state, intRule, a, p)
	+ complexQuad2D(integrateGeneratorMt_RAD, &state, intRule, p, b);
    } else {
      return complexQuad2D(integrateGeneratorMt_RAD, &state, intRule, a, b);
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

std::complex<float> integrateGeneratorN_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;

  pS->r = x.x;
  pS->z = x.y;

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

std::complex<float> integrateGeneratorNpOn_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;

  pS->r = x.x;
  pS->z = x.y;

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

std::complex<float> integrateGeneratorN0pOn_RAD(Float2 x, void *pState) {
  RadIntL * pS = (RadIntL *) pState;

  pS->r = x.x;
  pS->z = x.y;

  return complexLineIntegral(integrateSemiCircleN0pOn_RAD, pS, pS->semiCircleRule) * pS->r / static_cast<float>(2.0 * M_PI);
}

std::complex<float> complexConeIntegral(std::complex<float>(*integrand)(Float2, void*), void* state, IntRule1D intRule,
				  Float2 start, Float2 end, int nSections) {
  Float2 delta = 1.0f/nSections * (end - start);
  std::complex<float> sum = 0.0f;
  for (int s = 0; s < nSections; ++s) {
    Float2 segmentStart = start + static_cast<float>(s) * delta;
    Float2 segmentEnd   = start + static_cast<float>(s+1) * delta;
    sum += complexQuad2D(integrand, state, intRule, segmentStart, segmentEnd);
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
	+ complexQuad2D(integrateGeneratorNpOn_RAD, &state, intRule, a, p)
	+ complexQuad2D(integrateGeneratorNpOn_RAD, &state, intRule, p, b);
    } else {
      return complexQuad2D(integrateGeneratorN_RAD, &state, intRule, a, b);
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
