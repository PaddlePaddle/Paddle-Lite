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
using namespace metal;

kernel void batch_norm_relu_3x3(texture2d_array<float, access::sample> inTexture[[texture(0)]],
    texture2d_array<float, access::write> outTexture[[texture(1)]],
    const device float4* new_scale[[buffer(0)]],
    const device float4* new_biase[[buffer(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }

    float4 input;
    float4 output;
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    input = inTexture.sample(sample, gid.x, gid.y, gid.z);
    output = fmax(input * new_scale[gid.z] + new_biase[gid.z], 0.0);
    outTexture.write(output, gid.xy, gid.z);
}
