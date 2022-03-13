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
    int iW;
    int iH;
    int oW;
    int oH;
    int isize;
    int osize;
    int oarraysize;
    int start[4];
    int endC;
};

kernel void slice(device ftype* input[[buffer(0)]],
    device ftype* output[[buffer(1)]],
    constant MetalSliceParam& param[[buffer(2)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= param.oW || gid.y >= param.oH || gid.z >= param.oarraysize) {
        return;
    }
    for (int i = param.start[1], j = 0; i < param.endC; i++, j++) {
        int in_idx =
            i * param.isize + (param.start[2] + gid.y) * param.iW + (param.start[3] + gid.x);
        int out_idx = j * param.osize + gid.y * param.oW + gid.x;
        output[out_idx] = input[in_idx];
    }
}
