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

#include "Common.metal"
#include <metal_stdlib>

using namespace metal;

kernel void reduce_max_c(texture2d_array<ftype, access::read> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;

#if LITE_WITH_METAL_FULL
    float omax = FLT_MIN;
#else
    half omax = HALF_MIN;
#endif
    uint iAL = inTexture.get_array_size();
    for (uint i = 0; i < iAL; ++i) {
        ftype4 in = inTexture.read(uint2(gid.x, gid.y), i);
        omax = max(omax, in.x);
        omax = max(omax, in.y);
        omax = max(omax, in.z);
        omax = max(omax, in.w);
    }
    outTexture.write(ftype4(omax, 0.0, 0.0, 0.0), gid.xy, 0);
}

kernel void reduce_min_c(texture2d_array<ftype, access::read> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;

#if LITE_WITH_METAL_FULL
    float omin = FLT_MAX;
#else
    half omin = HALF_MAX;
#endif

    uint iAL = inTexture.get_array_size();
    for (uint i = 0; i < iAL - 1; ++i) {
        ftype4 in = inTexture.read(uint2(gid.x, gid.y), i);
        omin = min(omin, in.x);
        omin = min(omin, in.y);
        omin = min(omin, in.z);
        omin = min(omin, in.w);
    }
    ftype4 in_ = inTexture.read(uint2(gid.x, gid.y), iAL - 1);
    omin = abs(in_.x <= 1e-6) ? omin : min(omin, in_.x);
    omin = abs(in_.y <= 1e-6) ? omin : min(omin, in_.y);
    omin = abs(in_.z <= 1e-6) ? omin : min(omin, in_.z);
    omin = abs(in_.w <= 1e-6) ? omin : min(omin, in_.w);
    outTexture.write(ftype4(omin, 0.0, 0.0, 0.0), gid.xy, 0);
}

kernel void reduce_mean_c(texture2d_array<ftype, access::read> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;

#if LITE_WITH_METAL_FULL
    float omean = 0;
#else
    half omean = 0;
#endif
    uint iAL = inTexture.get_array_size();
    uint count = 4 * (iAL - 1);
    for (uint i = 0; i < iAL; ++i) {
        ftype4 in = inTexture.read(uint2(gid.x, gid.y), i);
        omean += in.x;
        omean += in.y;
        omean += in.z;
        omean += in.w;
    }
    ftype4 in_ = inTexture.read(uint2(gid.x, gid.y), iAL - 1);
    count = abs(in_.x <= 1e-6) ? count : count + 1;
    count = abs(in_.y <= 1e-6) ? count : count + 1;
    count = abs(in_.z <= 1e-6) ? count : count + 1;
    count = abs(in_.w <= 1e-6) ? count : count + 1;
    omean = omean / count;
    outTexture.write(ftype4(omean, 0.0, 0.0, 0.0), gid.xy, 0);
}

kernel void reduce_sum_c(texture2d_array<ftype, access::read> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;

#if LITE_WITH_METAL_FULL
    float osum = 0;
#else
    half osum = 0;
#endif
    uint iAL = inTexture.get_array_size();
    for (uint i = 0; i < iAL; ++i) {
        ftype4 in = inTexture.read(uint2(gid.x, gid.y), i);
        osum += in.x;
        osum += in.y;
        osum += in.z;
        osum += in.w;
    }
    outTexture.write(ftype4(osum, 0.0, 0.0, 0.0), gid.xy, 0);
}

kernel void reduce_mean_ch_unkeep(texture2d_array<ftype, access::read> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    constant RankParam& params[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;

    uint iC = params.idim[1];
    uint iH = params.idim[2];
    uint iAL = inTexture.get_array_size();

    ftype out[4] = {0.0};
    // out.xyzw
    for (uint k = 0; k < 4; k++) {
        // C direction average
        for (uint i = 0; i < iAL; i++) {
            ftype4 mean_h = 0.0;
            // H direction average
            for (uint j = 0; j < iH; j++) {
                ftype4 in_tex_h = inTexture.read(uint2(gid.z * 4 + k, j), i);
                mean_h += in_tex_h;
            }
            mean_h /= iH;

            out[k] += mean_h.x;
            out[k] += mean_h.y;
            out[k] += mean_h.z;
            out[k] += mean_h.w;
        }
        out[k] /= iC;
    }

    outTexture.write(ftype4(out[0], out[1], out[2], out[3]), gid.xy, gid.z);
}

kernel void reduce_mean_hw(texture2d_array<ftype, access::read> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    constant RankParam& params[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;

    uint iH = params.idim[2];
    uint iW = params.idim[3];

    ftype4 mean = 0.0;
    // H direction average
    for (uint i = 0; i < iH; i++) {
        // W direction average
        for (uint j = 0; j < iW; j++) {
            ftype4 in_tex = inTexture.read(uint2(j, i), gid.z);
            mean += in_tex;
        }
    }
    mean /= (iH * iW);
    outTexture.write(mean, gid.xy, gid.z);
}
