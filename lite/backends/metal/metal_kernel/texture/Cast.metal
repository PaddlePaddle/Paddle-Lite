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

enum CastType : int32_t {
    BOOL = 0,
    INT16 = 1,
    INT32 = 2,
    INT64 = 3,
    FP16 = 4,
    FP32 = 5,
    FP64 = 6,
};

struct CastParam {
    CastType inType;
    CastType outType;
};

kernel void cast(texture2d_array<ftype, access::read> input[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    constant CastParam& pm[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;

    ftype4 in = input.read(gid.xy, gid.z);
    ftype4 out = ftype4(0.0);

    switch (pm.inType) {
        case BOOL:
        case INT16:
        case INT32:
        case FP16:
        case FP32:
            out = in;
            break;
        case FP64:
            break;
        case INT64:
            break;
    }

    outTexture.write(out, gid.xy, gid.z);
}
