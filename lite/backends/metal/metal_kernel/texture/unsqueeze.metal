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

kernel void unsqueeze(texture2d_array<ftype, access::read> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    constant RankParam& params[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }

    // in: CPU-NCHW as GPU-NHWC
    // out: CPU-NCHW to GPU-NHWC
    if (params.irank < 4 && params.orank == 4) {
        ftype4 in = inTexture.read(uint2(gid.z, gid.y), gid.x / 4);
        uint idx = gid.x % 4;
        if (idx == 0) {
            outTexture.write(ftype4(in.x, 0.0, 0.0, 0.0), gid.xy, gid.z);
        } else if (idx == 1) {
            outTexture.write(ftype4(in.y, 0.0, 0.0, 0.0), gid.xy, gid.z);
        } else if (idx == 2) {
            outTexture.write(ftype4(in.z, 0.0, 0.0, 0.0), gid.xy, gid.z);
        } else if (idx == 3) {
            outTexture.write(ftype4(in.w, 0.0, 0.0, 0.0), gid.xy, gid.z);
        }
    } else {
        ftype4 out = inTexture.read(gid.xy, gid.z);
        outTexture.write(out, gid.xy, gid.z);
    }
}
