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

struct ArgParam {
    int orank;
};

inline int max_index(texture2d_array<ftype, access::read> inTexture[[texture(0)]], uint2 gid) {
    int index = 0;
#if LITE_WITH_METAL_FULL
    float omax = -FLT_MAX;
#else
    half omax = -HALF_MAX;
#endif
    uint iAL = inTexture.get_array_size();
    for (uint i = 0; i < iAL; i++) {
        ftype4 in = inTexture.read(uint2(gid.x, gid.y), i);
        if (omax < in.r) {
            omax = in.r;
            index = i * 4 + 0;
        }
        if (omax < in.g) {
            omax = in.g;
            index = i * 4 + 1;
        }
        if (omax < in.b) {
            omax = in.b;
            index = i * 4 + 2;
        }
        if (omax < in.a) {
            omax = in.a;
            index = i * 4 + 3;
        }
    }
    return index;
}

kernel void arg_max_c(texture2d_array<ftype, access::read> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    constant ArgParam& param[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;

    // dimensions = 4, CPU is NCHW, GPU is NHWC
    if (param.orank == 4) {
        int index = max_index(inTexture, gid.xy);
        outTexture.write(ftype4(index, 0.0, 0.0, 0.0), gid.xy, gid.z);
    }
    // dimensions < 4, CPU is NCHW, GPU treat as NHWC
    else {
        uint ix = gid.z * 4;
        uint iy = gid.x;

        int index_r = max_index(inTexture, uint2(ix, iy));
        int index_g = max_index(inTexture, uint2(ix + 1, iy));
        int index_b = max_index(inTexture, uint2(ix + 2, iy));
        int index_a = max_index(inTexture, uint2(ix + 3, iy));

        outTexture.write(ftype4(index_r, index_g, index_b, index_a), gid.xy, gid.z);
    }
}

kernel void arg_max_h(texture2d_array<ftype, access::read> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    constant ArgParam& param[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;

#if LITE_WITH_METAL_FULL
    float omax = -FLT_MAX;
#else
    half omax = -HALF_MAX;
#endif

    if (param.orank == 4) {
        uint iAL = inTexture.get_height();
        ftype4 cur_data = omax;
        ftype4 max_data = omax;
        uint4 cur_idx = 0;
        uint4 max_idx = 0;
        bool4 flag_v = false;

        for (uint i = 0; i < iAL; i++) {
            cur_data = inTexture.read(uint2(gid.x, i), gid.z);
            cur_idx = uint4(i);
            flag_v = bool4(max_data >= cur_data);
            max_data = select(cur_data, max_data, flag_v);
            max_idx = select(cur_idx, max_idx, flag_v);
        }
        outTexture.write(ftype4(max_idx), gid.xy, gid.z);
    }
}

kernel void arg_max_w(texture2d_array<ftype, access::read> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    constant ArgParam& param[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;

#if LITE_WITH_METAL_FULL
    float omax = -FLT_MAX;
#else
    half omax = -HALF_MAX;
#endif

    if (param.orank == 4) {
        uint iAL = inTexture.get_width();
        ftype4 cur_data = omax;
        ftype4 max_data = omax;
        uint4 cur_idx = 0;
        uint4 max_idx = 0;
        bool4 flag_v = false;

        for (uint i = 0; i < iAL; i++) {
            cur_data = inTexture.read(uint2(i, gid.y), gid.z);
            cur_idx = uint4(i);
            flag_v = bool4(max_data >= cur_data);
            max_data = select(cur_data, max_data, flag_v);
            max_idx = select(cur_idx, max_idx, flag_v);
        }
        outTexture.write(ftype4(max_idx), gid.xy, gid.z);
    }
}