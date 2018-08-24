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
using namespace metal;



kernel void prelu_channel(texture2d_array<float, access::sample> inTexture [[texture(0)]],
                           texture2d_array<float, access::write> outTexture [[texture(1)]],
                           const device float4 *alpha [[buffer(0)]],
                           uint3 gid [[thread_position_in_grid]]){
  if (gid.x >= outTexture.get_width() ||
      gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size()) {
    return;
  }
  
  constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
  float4 input = inTexture.sample(sample, gid.x, gid.y, gid.z);
  float4 output;
  output.x = input.x > 0 ? input.x : alpha[gid.z].x;
  output.x = input.y > 0 ? input.y : alpha[gid.z].y;
  output.x = input.z > 0 ? input.z : alpha[gid.z].z;
  output.x = input.w > 0 ? input.w : alpha[gid.z].w;
  outTexture.write(output, gid.xy, gid.z);
}


kernel void prelu_element(texture2d_array<float, access::sample> inTexture [[texture(0)]],
                          texture2d_array<float, access::write> outTexture [[texture(1)]],
                          const device float4 *alpha [[buffer(0)]],
                          uint3 gid [[thread_position_in_grid]]){
  if (gid.x >= outTexture.get_width() ||
      gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size()) {
    return;
  }
  
  constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
  float4 input = inTexture.sample(sample, gid.x, gid.y, gid.z);
  
  int alpha_to = (gid.y * inTexture.get_width() + gid.x) * inTexture.get_array_size();
  
  float4 output;
  output.x = input.x > 0 ? input.x : alpha[alpha_to + gid.z].x;
  output.x = input.y > 0 ? input.y : alpha[alpha_to + gid.z].y;
  output.x = input.z > 0 ? input.z : alpha[alpha_to + gid.z].z;
  output.x = input.w > 0 ? input.w : alpha[alpha_to + gid.z].w;
  outTexture.write(output, gid.xy, gid.z);
}


kernel void prelu_other(texture2d_array<float, access::sample> inTexture [[texture(0)]],
                          texture2d_array<float, access::write> outTexture [[texture(1)]],
                          const device float *alpha [[buffer(0)]],
                          uint3 gid [[thread_position_in_grid]]){
  if (gid.x >= outTexture.get_width() ||
      gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size()) {
    return;
  }
  
  constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
  float4 input = inTexture.sample(sample, gid.x, gid.y, gid.z);
  
  float4 output;
  output.x = input.x > 0 ? input.x : alpha[0];
  output.x = input.y > 0 ? input.y : alpha[0];
  output.x = input.z > 0 ? input.z : alpha[0];
  output.x = input.w > 0 ? input.w : alpha[0];
  outTexture.write(output, gid.xy, gid.z);
}
