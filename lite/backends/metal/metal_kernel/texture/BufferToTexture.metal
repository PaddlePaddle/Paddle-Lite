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

kernel void buffer_to_texture_kernel(const device float* input[[buffer(0)]],
    texture2d<float, access::write> outTexture[[texture(0)]],
    uint2 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) {
        return;
    }

    float y = input[outTexture.get_width() * gid.y + gid.x];
    outTexture.write(float4(y, 0.0f, 0.0f, 0.0f), gid);
}

kernel void buffer_to_texture_kernel_half(const device float* input[[buffer(0)]],
    texture2d<half, access::write> outTexture[[texture(0)]],
    uint2 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) {
        return;
    }

    float y = input[outTexture.get_width() * gid.y + gid.x];
    outTexture.write(half4(y, 0.0f, 0.0f, 0.0f), gid);
}

kernel void buffer_to_texture_kernel_channel_3(const device float* input[[buffer(0)]],
    texture2d<float, access::write> outTexture[[texture(0)]],
    uint2 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) {
        return;
    }

    int offset = outTexture.get_width() * outTexture.get_height();
    float y0 = input[outTexture.get_width() * gid.y + gid.x + 0 * offset];
    float y1 = input[outTexture.get_width() * gid.y + gid.x + 1 * offset];
    float y2 = input[outTexture.get_width() * gid.y + gid.x + 2 * offset];
    outTexture.write(float4(y0, y1, y2, 0.0f), gid);
}

kernel void buffer_to_texture_kernel_half_channel_3(const device float* input[[buffer(0)]],
    texture2d<half, access::write> outTexture[[texture(0)]],
    uint2 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) {
        return;
    }

    int offset = outTexture.get_width() * outTexture.get_height();
    float y0 = input[outTexture.get_width() * gid.y + gid.x + 0 * offset];
    float y1 = input[outTexture.get_width() * gid.y + gid.x + 1 * offset];
    float y2 = input[outTexture.get_width() * gid.y + gid.x + 2 * offset];
    outTexture.write(half4(y0, y1, y2, 0.0f), gid);
}
