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

struct TransposeParam {
  int iC;
  int oC;
  int axis[4];
};

kernel void transpose_copy_float(
    texture2d_array<float, access::read> inTexture [[texture(0)]],
    texture2d_array<float, access::write> outTexture [[texture(1)]],
    constant TransposeParam &pm [[buffer(0)]],
    uint3 gid [[thread_position_in_grid]]) {
  outTexture.write(inTexture.read(gid.xy, gid.z), gid.xy, gid.z);
}
kernel void transpose_copy_half(texture2d_array<half, access::read> inTexture
                                [[texture(0)]],
                                texture2d_array<half, access::write> outTexture
                                [[texture(1)]],
                                constant TransposeParam &pm [[buffer(0)]],
                                uint3 gid [[thread_position_in_grid]]) {
  outTexture.write(inTexture.read(gid.xy, gid.z), gid.xy, gid.z);
}

#define R 4
#define P float
#include "TransposeKernel.inc.metal"
#undef P
#define P half
#include "TransposeKernel.inc.metal"
#undef P
#undef R

#define R 3
#define P float
#include "TransposeKernel.inc.metal"
#undef P
#define P half
#include "TransposeKernel.inc.metal"
#undef P
#undef R

#define R 2
#define P float
#include "TransposeKernel.inc.metal"
#undef P
#define P half
#include "TransposeKernel.inc.metal"
#undef P
#undef R
