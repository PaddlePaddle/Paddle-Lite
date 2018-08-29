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

kernel void concat(texture2d_array<float, access::read> in0 [[texture(0)]],
                   texture2d_array<float, access::read> in1 [[texture(1)]],
                   texture2d_array<float, access::read> in2 [[texture(2)]],
                   texture2d_array<float, access::read> in3 [[texture(3)]],
                   texture2d_array<float, access::read> in4 [[texture(4)]],
                   texture2d_array<float, access::read> in5 [[texture(5)]],
                   texture2d_array<float, access::read> inx [[texture(6)]],
                   texture2d_array<float, access::write> out [[texture(7)]],
                   constant ConcatParam & pm [[buffer(0)]],
                   uint3 gid [[thread_position_in_grid]]) {
  ConcatParam cp = pm;
  int xyzn[4] = {int(gid.x), int(gid.y), int(gid.z), 0}, abcd[4], oxyzn[4];
  float4 r;
  for (int i = 0; i < 4; i++) {
    xyzn[3] = i;
    xyzn2abcd(cp.odim[3], xyzn, abcd);
    int k = abcd[cp.axis] - cp.offset;
    int j = 0;
    if (k < 0) {
      r[i] = inx.read(gid.xy, gid.z)[i];
    } else {
      for (; j < 6; j++) {
        if (k < cp.vdim[j]) {
          break;
        }
        k -= cp.vdim[j];
      }
      int ta = cp.odim[cp.axis];
      abcd[cp.axis] = k;
      cp.odim[cp.axis] = cp.vdim[j];
      abcd2xyzn(cp.odim[3], abcd, oxyzn);
      cp.odim[cp.axis] = ta;
      switch (j) {
        case 0: r[i] = in0.read(uint2(oxyzn[0], oxyzn[1]), oxyzn[2])[oxyzn[3]]; break;
        case 1: r[i] = in1.read(uint2(oxyzn[0], oxyzn[1]), oxyzn[2])[oxyzn[3]]; break;
        case 2: r[i] = in2.read(uint2(oxyzn[0], oxyzn[1]), oxyzn[2])[oxyzn[3]]; break;
        case 3: r[i] = in3.read(uint2(oxyzn[0], oxyzn[1]), oxyzn[2])[oxyzn[3]]; break;
        case 4: r[i] = in4.read(uint2(oxyzn[0], oxyzn[1]), oxyzn[2])[oxyzn[3]]; break;
        case 5: r[i] = in5.read(uint2(oxyzn[0], oxyzn[1]), oxyzn[2])[oxyzn[3]]; break;
      }
    }
  }
  out.write(r, gid.xy, gid.z);
}
