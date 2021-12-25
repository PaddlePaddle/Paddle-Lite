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

kernel void pixel_shuffle(texture2d_array<float, access::sample> inTexture[[texture(0)]],
    texture2d_array<float, access::write> outTexture[[texture(1)]],
    constant PixelShuffleParam& param[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);

    int upscale_factor = param.upscale_factor;
    int inX = gid.x / upscale_factor;
    int inY = gid.y / upscale_factor;

    float4 res;
    for (int i = 0; i < 4; i++) {
        int c = gid.z * 4 + i;
        int inC = c * upscale_factor * upscale_factor + (gid.y % upscale_factor) * upscale_factor +
                  gid.x % upscale_factor;
        float4 input = inTexture.read(uint2(inX, inY), inC / 4);
        res[i] = input[inC % 4];
    }

    outTexture.write(res, gid.xy, gid.z);
}

kernel void pixel_shuffle_half(texture2d_array<half, access::sample> inTexture[[texture(0)]],
    texture2d_array<half, access::write> outTexture[[texture(1)]],
    constant PixelShuffleParam& param[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);

    int upscale_factor = param.upscale_factor;
    int inX = gid.x / upscale_factor;
    int inY = gid.y / upscale_factor;

    half4 res;
    for (int i = 0; i < 4; i++) {
        int c = gid.z * 4 + i;
        int inC = c * upscale_factor * upscale_factor + (gid.y % upscale_factor) * upscale_factor +
                  gid.x % upscale_factor;
        half4 input = inTexture.read(uint2(inX, inY), inC / 4);
        res[i] = input[inC % 4];
    }

    outTexture.write(res, gid.xy, gid.z);
}
