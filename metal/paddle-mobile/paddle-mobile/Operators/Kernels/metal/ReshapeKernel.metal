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

struct ReshapeParam {
  int32_t idim[4];
  int32_t itrans[4];
  int32_t odim[4];
  int32_t otrans[4];
};

#define P float
#define DIN 4
#define DOUT 4
#include "ReshapeKernel.metal.inc"
#undef DOUT
#define DOUT 3
#include "ReshapeKernel.metal.inc"
#undef DOUT
#define DOUT 2
#include "ReshapeKernel.metal.inc"
#undef DOUT
#define DOUT 1
#include "ReshapeKernel.metal.inc"
#undef DOUT
#undef DIN

#define DIN 3
#define DOUT 4
#include "ReshapeKernel.metal.inc"
#undef DOUT
#define DOUT 3
#include "ReshapeKernel.metal.inc"
#undef DOUT
#define DOUT 2
#include "ReshapeKernel.metal.inc"
#undef DOUT
#define DOUT 1
#include "ReshapeKernel.metal.inc"
#undef DOUT
#undef DIN

#define DIN 2
#define DOUT 4
#include "ReshapeKernel.metal.inc"
#undef DOUT
#define DOUT 3
#include "ReshapeKernel.metal.inc"
#undef DOUT
#define DOUT 2
#include "ReshapeKernel.metal.inc"
#undef DOUT
#define DOUT 1
#include "ReshapeKernel.metal.inc"
#undef DOUT
#undef DIN

#define DIN 1
#define DOUT 4
#include "ReshapeKernel.metal.inc"
#undef DOUT
#define DOUT 3
#include "ReshapeKernel.metal.inc"
#undef DOUT
#define DOUT 2
#include "ReshapeKernel.metal.inc"
#undef DOUT
#define DOUT 1
#include "ReshapeKernel.metal.inc"
#undef DOUT
#undef DIN

#undef P

#define P half
#define DIN 4
#define DOUT 4
#include "ReshapeKernel.metal.inc"
#undef DOUT
#define DOUT 3
#include "ReshapeKernel.metal.inc"
#undef DOUT
#define DOUT 2
#include "ReshapeKernel.metal.inc"
#undef DOUT
#define DOUT 1
#include "ReshapeKernel.metal.inc"
#undef DOUT
#undef DIN

#define DIN 3
#define DOUT 4
#include "ReshapeKernel.metal.inc"
#undef DOUT
#define DOUT 3
#include "ReshapeKernel.metal.inc"
#undef DOUT
#define DOUT 2
#include "ReshapeKernel.metal.inc"
#undef DOUT
#define DOUT 1
#include "ReshapeKernel.metal.inc"
#undef DOUT
#undef DIN

#define DIN 2
#define DOUT 4
#include "ReshapeKernel.metal.inc"
#undef DOUT
#define DOUT 3
#include "ReshapeKernel.metal.inc"
#undef DOUT
#define DOUT 2
#include "ReshapeKernel.metal.inc"
#undef DOUT
#define DOUT 1
#include "ReshapeKernel.metal.inc"
#undef DOUT
#undef DIN

#define DIN 1
#define DOUT 4
#include "ReshapeKernel.metal.inc"
#undef DOUT
#define DOUT 3
#include "ReshapeKernel.metal.inc"
#undef DOUT
#define DOUT 2
#include "ReshapeKernel.metal.inc"
#undef DOUT
#define DOUT 1
#include "ReshapeKernel.metal.inc"
#undef DOUT
#undef DIN

#undef P
