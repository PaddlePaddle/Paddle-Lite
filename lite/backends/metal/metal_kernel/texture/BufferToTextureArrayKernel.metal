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

kernel void buf_to_tex(const device float *input [[buffer(0)]],
                       texture2d_array<ftype, access::write> outTexture
                       [[texture(0)]],
                       uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) {
    return;
  }

  float y = input[outTexture.get_width() * gid.y + gid.x];
  outTexture.write(ftype4(y, 0.0f, 0.0f, 0.0f), gid.xy, gid.z);
}

kernel void buf_to_tex_c_3(const device float *input [[buffer(0)]],
                           texture2d_array<ftype, access::write> outTexture
                           [[texture(0)]],
                           uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) {
    return;
  }

  int offset = outTexture.get_width() * outTexture.get_height();
  float y0 = input[outTexture.get_width() * gid.y + gid.x + 0 * offset];
  float y1 = input[outTexture.get_width() * gid.y + gid.x + 1 * offset];
  float y2 = input[outTexture.get_width() * gid.y + gid.x + 2 * offset];
  outTexture.write(ftype4(y0, y1, y2, 0.0f), gid.xy, gid.z);
}

kernel void buf_to_tex_c_n(const device float *input [[buffer(0)]],
                           texture2d_array<ftype, access::write> outTexture
                           [[texture(0)]],
                           uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size()) {
    return;
  }

  int cLength = outTexture.get_width() * outTexture.get_height();
  int inOffset = outTexture.get_width() * gid.y + gid.x;
  float y0 = input[inOffset + (gid.z * 4 + 0) * cLength];
  float y1 = input[inOffset + (gid.z * 4 + 1) * cLength];
  float y2 = input[inOffset + (gid.z * 4 + 2) * cLength];
  float y3 = input[inOffset + (gid.z * 4 + 3) * cLength];
  outTexture.write(ftype4(y0, y1, y2, y3), gid.xy, gid.z);
}

// half -> half
kernel void buf_h_to_tex_h(const device half *input [[buffer(0)]],
                           texture2d_array<half, access::write> outTexture [[texture(0)]],
                           uint3 gid [[thread_position_in_grid]]){
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height()) {
        return;
    }
    
    half y = input[outTexture.get_width() * gid.y + gid.x];
    outTexture.write(half4(y, 0.0f, 0.0f, 0.0f), gid.xy, gid.z);
}
