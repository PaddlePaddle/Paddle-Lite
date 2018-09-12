/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 
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

#define P float
#define R 4
#include "ConcatKernel.inc.metal"
#undef R
#define R 3
#include "ConcatKernel.inc.metal"
#undef R
#define R 2
#include "ConcatKernel.inc.metal"
#undef R
#define R 1
#include "ConcatKernel.inc.metal"
#undef R
#undef P

#define P half
#define R 4
#include "ConcatKernel.inc.metal"
#undef R
#define R 3
#include "ConcatKernel.inc.metal"
#undef R
#define R 2
#include "ConcatKernel.inc.metal"
#undef R
#define R 1
#include "ConcatKernel.inc.metal"
#undef R
#undef P
