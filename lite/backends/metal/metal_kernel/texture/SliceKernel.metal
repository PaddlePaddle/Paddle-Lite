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

struct MetalSliceParam {
    short start0;
    short start1;
    short start2;
    short start3;
    short end0;
    short end1;
    short end2;
    short end3;
    int iC;
    int oC;
};

kernel void slice(texture2d_array<ftype, access::sample> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    constant MetalSliceParam& param[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    ftype4 output;
    for (int i = 0; i < 4; ++i) {
        int tmp = gid.z * 4 + i;
        int output_c = tmp % param.oC;
        int output_n = tmp / param.oC;
        int c = output_c + param.start1;
        tmp = output_n * param.iC + c;
        int input_z = tmp / 4;
        int input_c = tmp % 4;
        const ftype4 input = inTexture.read(gid.xy, input_z);
        output[i] = input[input_c % 4];
    }
    outTexture.write(output, gid.xy, gid.z);
}
