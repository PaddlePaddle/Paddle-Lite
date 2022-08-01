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

struct FeedParam {
    int32_t isize;
    int32_t idim[4];
};

using namespace metal;

kernel void buf_to_tex_c_n(const device float* input[[buffer(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(0)]],
    constant FeedParam& param[[buffer(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }

    int inMax = param.idim[0] * param.idim[1] * param.idim[2] * param.idim[3];
    int page = outTexture.get_width() * outTexture.get_height();
    int offset = outTexture.get_width() * gid.y + gid.x;

    float4 output = float4(0.0);
    for (int i = 0; i < 4; i++) {
        int index = offset + (gid.z * 4 + i) * page;
        if (index < inMax) {
            output[i] = input[index];
        }
    }
    outTexture.write(ftype4(output.x, output.y, output.z, output.w), gid.xy, gid.z);
}

// half -> half
kernel void buf_h_to_tex_h(const device half* input[[buffer(0)]],
    texture2d_array<half, access::write> outTexture[[texture(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }
    int gidz = gid.z * 4;
    int output_size = outTexture.get_width() * outTexture.get_height();
    int output_index = outTexture.get_width() * gid.y + gid.x;

    half y0 = input[(gidz)*output_size + output_index];
    half y1 = input[(gidz + 1) * output_size + output_index];
    half y2 = input[(gidz + 2) * output_size + output_index];
    half y3 = input[(gidz + 3) * output_size + output_index];

    outTexture.write(half4(y0, y1, y2, y3), gid.xy, gid.z);
}
