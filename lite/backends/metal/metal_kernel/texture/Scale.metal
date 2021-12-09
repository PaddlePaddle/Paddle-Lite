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

kernel void scale(texture2d<float, access::sample> inTexture[[texture(0)]],
    texture2d<float, access::write> outTexture[[texture(1)]],
    uint2 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;
    float w_stride = inTexture.get_width() / outTexture.get_width();
    float h_stride = inTexture.get_height() / outTexture.get_height();
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    float4 input = inTexture.sample(sample, float2(gid.x * w_stride, gid.y * h_stride), 0);
    outTexture.write(input, gid);
}

kernel void scale_half(texture2d<float, access::sample> inTexture[[texture(0)]],
    texture2d<half, access::write> outTexture[[texture(1)]],
    uint2 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;
    float w_stride = inTexture.get_width() / outTexture.get_width();
    float h_stride = inTexture.get_height() / outTexture.get_height();
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    float4 input = inTexture.sample(sample, float2(gid.x * w_stride, gid.y * h_stride), 0);
    outTexture.write(half4(input), gid);
}
