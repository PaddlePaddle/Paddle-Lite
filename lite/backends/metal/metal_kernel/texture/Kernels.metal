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

kernel void place_holder(texture2d<half, access::read> inTexture[[texture(0)]],
    texture2d_array<half, access::write> outTexture[[texture(1)]],
    uint3 gid[[thread_position_in_grid]]) {
}

struct OutputDim {
    ushort width;
    ushort height;
    ushort strideX;
    ushort strideY;
};

kernel void resize(texture2d<half, access::read> inTexture[[texture(0)]],
    texture2d_array<half, access::write> outTexture[[texture(1)]],
    constant OutputDim& params[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;

    const uint2 pos = gid.xy * uint2(params.strideX, params.strideY);
    const half4 input = inTexture.read(pos);
    outTexture.write(half4(input.x, input.y, input.z, input.w), gid.xy, gid.z);
}

kernel void texture2d_to_2d_array(texture2d<ftype, access::read> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height()) {
        return;
    }
    const ftype4 input = inTexture.read(gid.xy);
    outTexture.write(input, gid.xy, 0);
}

kernel void texture2d_int_to_2d_array(texture2d<uint, access::read> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height()) {
        return;
    }
    const ftype4 input = static_cast<ftype4>(inTexture.read(gid.xy));
    outTexture.write(input, gid.xy, 0);
}

// texture2d_array -> buffer
kernel void tex2d_ary_to_buf(texture2d_array<ftype, access::read> input[[texture(0)]],
    device ftype* output[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    uint input_width = input.get_width();
    uint input_height = input.get_height();
    if (gid.x >= input_width || gid.y >= input_height || gid.z >= input.get_array_size()) {
        return;
    }

    const ftype4 value = input.read(gid.xy, gid.z);
    uint delta = input_width * input_height;
    uint output_to = 4 * gid.z * delta + gid.y * input_width + gid.x;

    output[output_to] = value.x;
    output[output_to + delta] = value.y;
    output[output_to + 2 * delta] = value.z;
    output[output_to + 3 * delta] = value.w;
}

kernel void tex2d_c1_to_c4(texture2d_array<ftype, access::read> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height() ||
        gid.z >= inTexture.get_array_size()) {
        return;
    }

    const ftype4 in = inTexture.read(gid.xy, gid.z);
    ftype4 out = ftype4(in.r, 0.0, 0.0, 0.0);
    outTexture.write(out, gid.xy, 0);
}
