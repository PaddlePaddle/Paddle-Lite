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

kernel void nms_fetch_result(texture2d_array<float, access::read> inTexture[[texture(0)]],
    device float* output[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height() ||
        gid.z >= inTexture.get_array_size()) {
        return;
    }

    int input_width = inTexture.get_width();
    const float4 input = inTexture.read(gid.xy, gid.z);
    output[gid.y * input_width + gid.x] = input.x;
}

kernel void nms_fetch_result_half(texture2d_array<half, access::read> inTexture[[texture(0)]],
    device float* output[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height() ||
        gid.z >= inTexture.get_array_size()) {
        return;
    }

    int input_width = inTexture.get_width();
    const half4 input = inTexture.read(gid.xy, gid.z);
    output[gid.y * input_width + gid.x] = input.x;
}

kernel void nms_fetch_bbox(texture2d_array<float, access::read> inTexture[[texture(0)]],
    device float4* output[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height() ||
        gid.z >= inTexture.get_array_size()) {
        return;
    }

    int input_width = inTexture.get_width();
    //  int input_height = inTexture.get_height();
    const float4 input = inTexture.read(gid.xy, gid.z);
    output[gid.y * input_width + gid.x] = input;
}

kernel void nms_fetch_bbox_half(texture2d_array<half, access::read> inTexture[[texture(0)]],
    device float4* output[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height() ||
        gid.z >= inTexture.get_array_size()) {
        return;
    }

    int input_width = inTexture.get_width();
    //  int input_height = inTexture.get_height();
    const half4 input = inTexture.read(gid.xy, gid.z);
    output[gid.y * input_width + gid.x] = float4(input);
}
