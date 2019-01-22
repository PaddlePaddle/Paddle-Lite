/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
 http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONRITIONS OF ANY KINR, either express or implied.
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
#define RIN 4
#define ROUT 4
#include "ReshapeKernel.inc.metal"
#undef ROUT
#define ROUT 3
#include "ReshapeKernel.inc.metal"
#undef ROUT
#define ROUT 2
#include "ReshapeKernel.inc.metal"
#undef ROUT
#define ROUT 1
#include "ReshapeKernel.inc.metal"
#undef ROUT
#undef RIN

#define RIN 3
#define ROUT 4
#include "ReshapeKernel.inc.metal"
#undef ROUT
#define ROUT 3
#include "ReshapeKernel.inc.metal"
#undef ROUT
#define ROUT 2
#include "ReshapeKernel.inc.metal"
#undef ROUT
#define ROUT 1
#include "ReshapeKernel.inc.metal"
#undef ROUT
#undef RIN

#define RIN 2
#define ROUT 4
#include "ReshapeKernel.inc.metal"
#undef ROUT
#define ROUT 3
#include "ReshapeKernel.inc.metal"
#undef ROUT
#define ROUT 2
#include "ReshapeKernel.inc.metal"
#undef ROUT
#define ROUT 1
#include "ReshapeKernel.inc.metal"
#undef ROUT
#undef RIN

#define RIN 1
#define ROUT 4
#include "ReshapeKernel.inc.metal"
#undef ROUT
#define ROUT 3
#include "ReshapeKernel.inc.metal"
#undef ROUT
#define ROUT 2
#include "ReshapeKernel.inc.metal"
#undef ROUT
#define ROUT 1
#include "ReshapeKernel.inc.metal"
#undef ROUT
#undef RIN

#undef P

#define P half
#define RIN 4
#define ROUT 4
#include "ReshapeKernel.inc.metal"
#undef ROUT
#define ROUT 3
#include "ReshapeKernel.inc.metal"
#undef ROUT
#define ROUT 2
#include "ReshapeKernel.inc.metal"
#undef ROUT
#define ROUT 1
#include "ReshapeKernel.inc.metal"
#undef ROUT
#undef RIN

#define RIN 3
#define ROUT 4
#include "ReshapeKernel.inc.metal"
#undef ROUT
#define ROUT 3
#include "ReshapeKernel.inc.metal"
#undef ROUT
#define ROUT 2
#include "ReshapeKernel.inc.metal"
#undef ROUT
#define ROUT 1
#include "ReshapeKernel.inc.metal"
#undef ROUT
#undef RIN

#define RIN 2
#define ROUT 4
#include "ReshapeKernel.inc.metal"
#undef ROUT
#define ROUT 3
#include "ReshapeKernel.inc.metal"
#undef ROUT
#define ROUT 2
#include "ReshapeKernel.inc.metal"
#undef ROUT
#define ROUT 1
#include "ReshapeKernel.inc.metal"
#undef ROUT
#undef RIN

#define RIN 1
#define ROUT 4
#include "ReshapeKernel.inc.metal"
#undef ROUT
#define ROUT 3
#include "ReshapeKernel.inc.metal"
#undef ROUT
#define ROUT 2
#include "ReshapeKernel.inc.metal"
#undef ROUT
#define ROUT 1
#include "ReshapeKernel.inc.metal"
#undef ROUT
#undef RIN
#undef P
