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

kernel void transpose(texture2d_array<float, access::read> inTexture [[texture(0)]],
                      texture2d_array<float, access::write> outTexture [[texture(1)]],
                      constant TransposeParam &pm [[buffer(0)]],
                      uint3 gid [[thread_position_in_grid]]) {
  
  
  if ((pm.axis[0] == 0) && (pm.axis[1] == 1) && (pm.axis[2] == 2) && (pm.axis[3] == 3)) {
    // do nothing
    float4 r = inTexture.read(gid.xy, gid.z);
    outTexture.write(r, gid.xy, gid.z);
  } else {
    float4 r;
    for (int n = 0; n < 4; n++) {
      int ixyzn[] = {int(gid.x), int(gid.y), int(gid.z), n};
      int iabcd[4], oabcd[4], oxyzn[4];
      xyzn2abcd(pm.oC, ixyzn, iabcd);
      oabcd[pm.axis[0]] = iabcd[0];
      oabcd[pm.axis[1]] = iabcd[1];
      oabcd[pm.axis[2]] = iabcd[2];
      oabcd[pm.axis[3]] = iabcd[3];
      abcd2xyzn(pm.iC, oabcd, oxyzn);
      float4 rt = inTexture.read(uint2(oxyzn[0], oxyzn[1]), oxyzn[2]);
      r[n] = rt[oxyzn[3]];
    }
    outTexture.write(r, gid.xy, gid.z);
  }
}

kernel void transpose_half(texture2d_array<half, access::read> inTexture [[texture(0)]],
                           texture2d_array<half, access::write> outTexture [[texture(1)]],
                           constant TransposeParam &pm [[buffer(0)]],
                           uint3 gid [[thread_position_in_grid]]) {
  
  
  if ((pm.axis[0] == 0) && (pm.axis[1] == 1) && (pm.axis[2] == 2) && (pm.axis[3] == 3)) {
    // do nothing
    half4 r = inTexture.read(gid.xy, gid.z);
    outTexture.write(r, gid.xy, gid.z);
  } else {
    half4 r;
    for (int n = 0; n < 4; n++) {
      int ixyzn[] = {int(gid.x), int(gid.y), int(gid.z), n};
      int iabcd[4], oabcd[4], oxyzn[4];
      xyzn2abcd(pm.oC, ixyzn, iabcd);
      oabcd[pm.axis[0]] = iabcd[0];
      oabcd[pm.axis[1]] = iabcd[1];
      oabcd[pm.axis[2]] = iabcd[2];
      oabcd[pm.axis[3]] = iabcd[3];
      abcd2xyzn(pm.iC, oabcd, oxyzn);
      half4 rt = inTexture.read(uint2(oxyzn[0], oxyzn[1]), oxyzn[2]);
      r[n] = rt[oxyzn[3]];
    }
    outTexture.write(r, gid.xy, gid.z);
  }
}

