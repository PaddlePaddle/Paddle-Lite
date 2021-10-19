/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License. */

#include <metal_stdlib>

#include "Common.metal"

using namespace metal;

struct ConcatParam {
    int32_t odim[4];
    int32_t axis;
    int32_t offset;
    int32_t trans[4];
    int32_t vdim[6];
};

#define VNORMAL 1
#define VX 2
#define VY 3
#define VZ 4

// R:input dim size N:input number V: direction

// >> normal mode (loop mode)

// >> fast mode
// only support concat_{2,3,4}_{2,3,4,5,6}_y_{float,half}
// only support concat_{3,4}_{2,3,4,5,6}_x_{float,half}
// only support concat_{1,2,3,4}_{2,3,4,5,6}_z_{float,half}

// >> special model
// lens: (R=4, N=3, V=normal)
// lens: (R=2, N=3, V=normal)
// lens: (R=2, N=2, V=normal)
// lens: (R=4, N=2, V=z)
// ssd-ar: (R=4, N=3, V=z)
// ssd-ar: (R=3, N=2, V=y)
// ssd-ar: (R=3, N=5, V=x)
// ssd-ar: (R=2, N=5, V=x)
// ssd: (R=2, N=6, V=y),
// ssd: (R=3, N=6, V=y)
// genet: (R=4, N=2, V=normal)
// gesture recognizing: (R=2, N=3, V=x)

#pragma mark -
#pragma mark normal

#define V VNORMAL
#define R 4
#define N 4
#define P ftype
#include "ConcatKernel.inc.metal"
#undef P
#undef N
#undef R
#undef V

#define V VNORMAL
#define R 4
#define N 3
#define P ftype
#include "ConcatKernel.inc.metal"
#undef P
#undef N
#undef R
#undef V

#define V VNORMAL
#define R 4
#define N 2
#define P ftype
#include "ConcatKernel.inc.metal"
#undef P
#undef N
#undef R
#undef V

#define V VNORMAL
#define R 3
#define N 3
#define P ftype
#include "ConcatKernel.inc.metal"
#undef P
#undef N
#undef R
#undef V

#define V VNORMAL
#define R 3
#define N 2
#define P ftype
#include "ConcatKernel.inc.metal"
#undef P
#undef N
#undef R
#undef V

#define V VNORMAL
#define R 2
#define N 3
#define P ftype
#include "ConcatKernel.inc.metal"
#undef P
#undef N
#undef R
#undef V

#define V VNORMAL
#define R 2
#define N 2
#define P ftype
#include "ConcatKernel.inc.metal"
#undef P
#undef N
#undef R
#undef V

#pragma mark -
#pragma mark z

#define V VZ
#define R 4
#define N 6
#define P ftype
#include "ConcatKernel.inc.metal"
#undef P
#undef N
#undef R
#undef V

#define V VZ
#define R 4
#define N 5
#define P ftype
#include "ConcatKernel.inc.metal"
#undef P
#undef N
#undef R
#undef V

#define V VZ
#define R 4
#define N 4
#define P ftype
#include "ConcatKernel.inc.metal"
#undef P
#undef N
#undef R
#undef V

#define V VZ
#define R 4
#define N 3
#define P ftype
#include "ConcatKernel.inc.metal"
#undef P
#undef N
#undef R
#undef V

#define V VZ
#define R 4
#define N 2
#define P ftype
#include "ConcatKernel.inc.metal"
#undef P
#undef N
#undef R
#undef V

#define V VZ
#define R 3
#define N 5
#define P ftype
#include "ConcatKernel.inc.metal"
#undef P
#undef N
#undef R
#undef V

#define V VZ
#define R 3
#define N 3
#define P ftype
#include "ConcatKernel.inc.metal"
#undef P
#undef N
#undef R
#undef V

#pragma mark -
#pragma mark x

#define V VX
#define R 3
#define N 6
#define P ftype
#include "ConcatKernel.inc.metal"
#undef P
#undef N
#undef R
#undef V

#define V VX
#define R 3
#define N 5
#define P ftype
#include "ConcatKernel.inc.metal"
#undef P
#undef N
#undef R
#undef V

#define V VX
#define R 3
#define N 4
#define P ftype
#include "ConcatKernel.inc.metal"
#undef P
#undef N
#undef R
#undef V

#define V VX
#define R 3
#define N 3
#define P ftype
#include "ConcatKernel.inc.metal"
#undef P
#undef N
#undef R
#undef V

#define V VX
#define R 3
#define N 2
#define P ftype
#include "ConcatKernel.inc.metal"
#undef P
#undef N
#undef R
#undef V

#define V VX
#define R 2
#define N 6
#define P ftype
#include "ConcatKernel.inc.metal"
#undef P
#undef N
#undef R
#undef V

#define V VX
#define R 2
#define N 5
#define P ftype
#include "ConcatKernel.inc.metal"
#undef P
#undef N
#undef R
#undef V

#define V VX
#define R 2
#define N 4
#define P ftype
#include "ConcatKernel.inc.metal"
#undef P
#undef N
#undef R
#undef V

#define V VX
#define R 2
#define N 3
#define P ftype
#include "ConcatKernel.inc.metal"
#undef P
#undef N
#undef R
#undef V

#pragma mark -
#pragma mark y

#define V VY
#define R 4
#define N 3
#define P ftype
#include "ConcatKernel.inc.metal"
#undef P
#undef N
#undef R
#undef V

#define V VY
#define R 3
#define N 6
#define P ftype
#include "ConcatKernel.inc.metal"
#undef P
#undef N
#undef R
#undef V

#define V VY
#define R 3
#define N 3
#define P ftype
#include "ConcatKernel.inc.metal"
#undef P
#undef N
#undef R
#undef V

#define V VY
#define R 3
#define N 2
#define P ftype
#include "ConcatKernel.inc.metal"
#undef P
#undef N
#undef R
#undef V

#define V VY
#define R 2
#define N 6
#define P ftype
#include "ConcatKernel.inc.metal"
#undef P
#undef N
#undef R
#undef V

#define V VY
#define R 2
#define N 5
#define P ftype
#include "ConcatKernel.inc.metal"
#undef P
#undef N
#undef R
#undef V

#define V VY
#define R 2
#define N 2
#define P ftype
#include "ConcatKernel.inc.metal"
#undef P
#undef N
#undef R
#undef V
