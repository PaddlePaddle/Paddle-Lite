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

enum CompareType : int32_t {
    Equal = 0,
    NotEqual = 1,
    LessThan = 2,
    LessEqual = 3,
    GreaterThan = 4,
    GreaterEqual = 5,
};

struct CompareParam {
    CompareType compareType;
};

kernel void compare(texture2d_array<ftype, access::read> inputX[[texture(0)]],
    texture2d_array<ftype, access::read> inputY[[texture(1)]],
    texture2d_array<ftype, access::write> outTexture[[texture(2)]],
    constant CompareParam& pm[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;

    ftype4 rx = inputX.read(gid.xy, gid.z);
    ftype4 ry = inputY.read(gid.xy, gid.z);
    ftype4 out = ftype4(0.0);

    switch (pm.compareType) {
        case Equal:
            out = ftype4((rx.x == ry.x) ? 1.0 : 0.0,
                (rx.y == ry.y) ? 1.0 : 0.0,
                (rx.z == ry.z) ? 1.0 : 0.0,
                (rx.w == ry.w) ? 1.0 : 0.0);
            break;
        case NotEqual:
            out = ftype4((rx.x != ry.x) ? 1.0 : 0.0,
                (rx.y != ry.y) ? 1.0 : 0.0,
                (rx.z != ry.z) ? 1.0 : 0.0,
                (rx.w != ry.w) ? 1.0 : 0.0);
            break;
        case LessThan:
            out = ftype4((rx.x < ry.x) ? 1.0 : 0.0,
                (rx.y < ry.y) ? 1.0 : 0.0,
                (rx.z < ry.z) ? 1.0 : 0.0,
                (rx.w < ry.w) ? 1.0 : 0.0);
            break;
        case LessEqual:
            out = ftype4((rx.x <= ry.x) ? 1.0 : 0.0,
                (rx.y <= ry.y) ? 1.0 : 0.0,
                (rx.z <= ry.z) ? 1.0 : 0.0,
                (rx.w <= ry.w) ? 1.0 : 0.0);
            break;
        case GreaterThan:
            out = ftype4((rx.x > ry.x) ? 1.0 : 0.0,
                (rx.y > ry.y) ? 1.0 : 0.0,
                (rx.z > ry.z) ? 1.0 : 0.0,
                (rx.w > ry.w) ? 1.0 : 0.0);
            break;
        case GreaterEqual:
            out = ftype4((rx.x >= ry.x) ? 1.0 : 0.0,
                (rx.y >= ry.y) ? 1.0 : 0.0,
                (rx.z >= ry.z) ? 1.0 : 0.0,
                (rx.w >= ry.w) ? 1.0 : 0.0);
            break;
    }

    outTexture.write(out, gid.xy, gid.z);
}
