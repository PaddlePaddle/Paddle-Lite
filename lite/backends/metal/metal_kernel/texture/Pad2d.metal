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

kernel void pad2d(texture2d_array<ftype, access::read> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    constant Pad2dParam& pm[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    int x = gid.x - pm.paddingLeft;
    int y = gid.y - pm.paddingTop;
    if (pm.mode == 0) {
        if (x < 0 || y < 0 || x >= inTexture.get_width() || y >= inTexture.get_height()) {
            outTexture.write(ftype4(pm.padValue), uint2(gid.xy), gid.z);
        } else {
            outTexture.write(inTexture.read(uint2(x, y), gid.z), uint2(gid.xy), gid.z);
        }
    } else if (pm.mode == 1) {
        x = abs(x);
        y = abs(y);
        uint w = inTexture.get_width();
        uint h = inTexture.get_height();
        x = x < w ? x : 2 * w - 2 - x;
        y = y < h ? y : 2 * h - 2 - y;
        outTexture.write(inTexture.read(uint2(x, y), gid.z), uint2(gid.xy), gid.z);
    } else if (pm.mode == 2) {
        uint w = inTexture.get_width();
        uint h = inTexture.get_height();
        x = x > 0 ? x : 0;
        x = x < w ? x : w - 1;
        y = y > 0 ? y : 0;
        y = y < h ? y : h - 1;
        outTexture.write(inTexture.read(uint2(x, y), gid.z), uint2(gid.xy), gid.z);
    }
}
