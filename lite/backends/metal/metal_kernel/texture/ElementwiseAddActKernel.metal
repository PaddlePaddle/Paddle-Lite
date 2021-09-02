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

#define P float

#define RELU relu
#define ACT_TYPE relu
#include "ElementwiseAddActKernel.inc.metal"
#undef ACT_TYPE
#undef RELU

#define PRELU_CHANNEL prelu_channel
#define ACT_TYPE channel
#include "ElementwiseAddActKernel.inc.metal"
#undef ACT_TYPE
#undef PRELU_CHANNEL

#define PRELU_ELEMENT element
#define ACT_TYPE prelu_element
#include "ElementwiseAddActKernel.inc.metal"
#undef ACT_TYPE
#undef PRELU_ELEMENT

#define PRELU_OTHER other
#define ACT_TYPE prelu_other
#include "ElementwiseAddActKernel.inc.metal"
#undef ACT_TYPE
#undef PRELU_OTHER

#undef P

#define P half

#define PRELU_CHANNEL channel
#define ACT_TYPE channel
#include "ElementwiseAddActKernel.inc.metal"
#undef ACT_TYPE
#undef PRELU_CHANNEL

#define PRELU_ELEMENT element
#define ACT_TYPE prelu_element
#include "ElementwiseAddActKernel.inc.metal"
#undef ACT_TYPE
#undef PRELU_ELEMENT

#define PRELU_OTHER other
#define ACT_TYPE prelu_other
#include "ElementwiseAddActKernel.inc.metal"
#undef ACT_TYPE
#undef PRELU_OTHER

#define RELU relu
#define ACT_TYPE relu
#include "ElementwiseAddActKernel.inc.metal"
#undef ACT_TYPE
#undef RELU

#undef P
