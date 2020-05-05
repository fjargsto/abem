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
#ifndef INTOPS_H
#define INTOPS_H

#include <stdbool.h>

/* ************************************************************************** */
/* Struct representing points in 2-dimensional Euclidean space. */
typedef struct {
  float x;
  float y;
} Float2;

/* Struct representing points in 3-dimensional Euclidean space. */
typedef struct {
  float x;
  float y;
  float z;
} Float3;

Float2 add2f(Float2 a, Float2 b);
Float3 add3f(Float3 a, Float3 b);

Float2 sub2f(Float2 a, Float2 b);
Float3 sub3f(Float3 a, Float3 b);

Float2 smul2f(float a, Float2 x);
Float3 smul3f(float a, Float3 x);

float dot2f(Float2 a, Float2 b);
float dot3f(Float3 a, Float3 b);

Float3 cross(Float3 a, Float3 b);

float norm2f(Float2 a);
float norm3f(Float3 a);

Float2 Normal2D(Float2 a, Float2 b);
Float3 Normal3D(Float3 a, Float3 b, Float3 c);

Float2 add2f(Float2 a, Float2 b);

typedef struct {
  float re;
  float im;
} Complex;

void Hankel1(int order, float x, Complex * pz);

void ComputeL_2D(float k, Float2 p, Float2 a, Float2 b, bool pOnElement, Complex * pResult);
void ComputeM_2D(float k, Float2 p, Float2 a, Float2 b, bool pOnElement, Complex * pResult);
void ComputeMt_2D(float k, Float2 p, Float2 normal_p, Float2 a, Float2 b, bool pOnElement, Complex * pResult);
void ComputeN_2D(float k, Float2 p, Float2 normal_p, Float2 a, Float2 b, bool pOnElement, Complex * pResult);

void ComputeL_RAD(float k, Float2 p, Float2 a, Float2 b, bool pOnElement, Complex * pResult);
void ComputeM_RAD(float k, Float2 p, Float2 a, Float2 b, bool pOnElement, Complex * pResult);
void ComputeMt_RAD(float k, Float2 p, Float2 vec_p, Float2 a, Float2 b, bool pOnElement, Complex * pResult);
void ComputeN_RAD(float k, Float2 p, Float2 vec_p, Float2 a, Float2 b, bool pOnElement, Complex * pResult);

void ComputeL_3D(float k, Float3 p, Float3 a, Float3 b, Float3 c, bool pOnElement, Complex * pResult);
void ComputeM_3D(float k, Float3 p, Float3 a, Float3 b, Float3 c, bool pOnElement, Complex * pResult);
void ComputeMt_3D(float k, Float3 p, Float3 vec_p, Float3 a, Float3 b, Float3 c, bool pOnElement, Complex * pResult);
void ComputeN_3D(float k, Float3 p, Float3 normal_p, Float3 a, Float3 b, Float3 c, bool pOnElement, Complex * pResult);

#endif /* INTOPS_H */