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

struct Relu6Param {
  float threshold;
};

struct LeakyReluParam {
  float alpha;
};

kernel void relu(texture2d_array<ftype, access::read> inTexture [[texture(0)]],
                 texture2d_array<ftype, access::write> outTexture
                 [[texture(1)]],
                 uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size())
    return;
  const ftype4 input = inTexture.read(gid.xy, gid.z);
  const ftype4 relu = fmax(input, 0.0);
  outTexture.write(relu, gid.xy, gid.z);
}

kernel void relu6(texture2d_array<ftype, access::read> inTexture [[texture(0)]],
                  texture2d_array<ftype, access::write> outTexture
                  [[texture(1)]],
                  constant Relu6Param &pm [[buffer(0)]],
                  uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size())
    return;
  const ftype4 input = inTexture.read(gid.xy, gid.z);
  const float threshold = pm.threshold;
  const ftype4 relu = fmin(fmax(input, 0.0), threshold);
  outTexture.write(relu, gid.xy, gid.z);
}

kernel void leaky_relu(texture2d_array<ftype, access::read> inTexture
                       [[texture(0)]],
                       texture2d_array<ftype, access::write> outTexture
                       [[texture(1)]],
                       constant LeakyReluParam &pm [[buffer(0)]],
                       uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size())
    return;
  const ftype4 input = inTexture.read(gid.xy, gid.z);
  const float alpha = pm.alpha;
  const ftype4 output = fmax(input, input * alpha);
  outTexture.write(output, gid.xy, gid.z);
}
