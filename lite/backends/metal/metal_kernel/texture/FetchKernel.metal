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

struct FetchParam {
    int32_t isize;
    int32_t idim[4];
};

kernel void fetch(texture2d_array<ftype, access::read> inTexture[[texture(0)]],
    device float* output[[buffer(0)]],
    constant FetchParam& param[[buffer(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    uint input_width = inTexture.get_width();
    uint input_height = inTexture.get_height();
    if (gid.x >= input_width || gid.y >= input_height || gid.z >= inTexture.get_array_size()) {
        return;
    }

    uint count = param.idim[0] * param.idim[1] * param.idim[2] * param.idim[3];
    //  dimensions == 4, data layout on CPU is NCHW, data layout on GPU is NHWC
    if (param.isize == 4) {
        const ftype4 input = inTexture.read(gid.xy, gid.z);
        uint delta = input_width * input_height;
        uint output_to = 4 * gid.z * delta + gid.y * input_width + gid.x;

        uint dst = output_to;
        if (dst < count) {
            output[dst] = input.x;
        }
        dst = output_to + 1 * delta;
        if (dst < count) {
            output[dst] = input.y;
        }
        dst = output_to + 2 * delta;
        if (dst < count) {
            output[dst] = input.z;
        }
        dst = output_to + 3 * delta;
        if (dst < count) {
            output[dst] = input.w;
        }
    }
    //  dimensions < 4, data layout on CPU is NCHW(texture width is H, height is W), data layout on
    //  GPU is NCHW
    // arraylength=(W+3)/4
    else {
        const ftype4 input = inTexture.read(gid.xy, gid.z);
        uint w = param.idim[2];
        uint c = param.idim[3];

        uint output_to = gid.y * c * w + gid.x * c + gid.z * 4;
        uint output_max = gid.y * c * w + gid.x * c + c;

        if (output_to < output_max) {
            output[output_to] = input.x;
        }

        output_to += 1;
        if (output_to < output_max) {
            output[output_to] = input.y;
        }

        output_to += 1;
        if (output_to < output_max) {
            output[output_to] = input.z;
        }

        output_to += 1;
        if (output_to < output_max) {
            output[output_to] = input.w;
        }
    }
}
