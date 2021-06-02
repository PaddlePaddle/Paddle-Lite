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

struct ScaleParam {
  float scale;
  float abias;
};

kernel void scale_before_bias_float(
    texture2d_array<float, access::read> inTexture [[texture(0)]],
    texture2d_array<float, access::write> outTexture [[texture(1)]],
    constant ScaleParam &pm [[buffer(0)]],
    uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size())
    return;
  constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
  const float4 input = inTexture.read(gid.xy, gid.z);
  const float scale = pm.scale;
  const float abias = pm.abias;
  const float4 output = scale * input + abias;
  outTexture.write(output, gid.xy, gid.z);
}

kernel void scale_after_bias_float(
    texture2d_array<float, access::read> inTexture [[texture(0)]],
    texture2d_array<float, access::write> outTexture [[texture(1)]],
    constant ScaleParam &pm [[buffer(0)]],
    uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size())
    return;
  constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
  const float4 input = inTexture.read(gid.xy, gid.z);
  const float scale = pm.scale;
  const float abias = pm.abias;
  const float4 output = scale * (input + abias);
  outTexture.write(output, gid.xy, gid.z);
}

kernel void scale_before_bias_half(
    texture2d_array<half, access::read> inTexture [[texture(0)]],
    texture2d_array<half, access::write> outTexture [[texture(1)]],
    constant ScaleParam &pm [[buffer(0)]],
    uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size())
    return;
  constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
  const half4 input = inTexture.read(gid.xy, gid.z);
  const float scale = pm.scale;
  const float abias = pm.abias;
  const float4 output = scale * (float4)input + abias;
  outTexture.write(half4(output), gid.xy, gid.z);
}

kernel void scale_after_bias_half(
    texture2d_array<half, access::read> inTexture [[texture(0)]],
    texture2d_array<half, access::write> outTexture [[texture(1)]],
    constant ScaleParam &pm [[buffer(0)]],
    uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size())
    return;
  constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
  const half4 input = inTexture.read(gid.xy, gid.z);
  const float scale = pm.scale;
  const float abias = pm.abias;
  const float4 output = scale * ((float4)input + abias);
  outTexture.write(half4(output), gid.xy, gid.z);
}
