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

kernel void mul(texture2d_array<ftype, access::sample> inputX[[texture(0)]],
    texture2d_array<ftype, access::sample> inputY[[texture(1)]],
    texture2d_array<ftype, access::write> outTexture[[texture(2)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    int xLen = inputX.get_width();
    ftype4 r = ftype4(0, 0, 0, 0);
    for (int i = 0; i < xLen; i++) {
        ftype4 iX = inputX.sample(sample, float2(i, gid.y), 0);
        ftype4 iY1 = inputY.sample(sample, float2(gid.x, i * 4), 0);
        ftype4 iY2 = inputY.sample(sample, float2(gid.x, i * 4 + 1), 0);
        ftype4 iY3 = inputY.sample(sample, float2(gid.x, i * 4 + 2), 0);
        ftype4 iY4 = inputY.sample(sample, float2(gid.x, i * 4 + 3), 0);
        ftype4 tY1 = ftype4(iY1.x, iY2.x, iY3.x, iY4.x);
        ftype4 tY2 = ftype4(iY1.y, iY2.y, iY3.y, iY4.y);
        ftype4 tY3 = ftype4(iY1.z, iY2.z, iY3.z, iY4.z);
        ftype4 tY4 = ftype4(iY1.w, iY2.w, iY3.w, iY4.w);
        r.x += dot(iX, tY1);
        r.y += dot(iX, tY2);
        r.z += dot(iX, tY3);
        r.w += dot(iX, tY4);
    }
    outTexture.write(r, gid.xy, gid.z);
}

kernel void mul_add(texture2d_array<ftype, access::sample> inputX[[texture(0)]],
    texture2d_array<ftype, access::sample> inputY[[texture(1)]],
    texture2d_array<ftype, access::sample> biasY[[texture(2)]],
    texture2d_array<ftype, access::write> outTexture[[texture(3)]],
    constant MetalActivationParam& param[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    int xLen = inputX.get_array_size();
    ftype4 r = ftype4(0, 0, 0, 0);
    for (int i = 0; i < xLen; i++) {
        ftype4 iX = inputX.sample(sample, float2(0, 0), i);
        ftype4 iY1 = inputY.sample(sample, float2(i * 4, 0), gid.z);
        ftype4 iY2 = inputY.sample(sample, float2(i * 4 + 1, 0), gid.z);
        ftype4 iY3 = inputY.sample(sample, float2(i * 4 + 2, 0), gid.z);
        ftype4 iY4 = inputY.sample(sample, float2(i * 4 + 3, 0), gid.z);
        ftype4 tY1 = ftype4(iY1.x, iY2.x, iY3.x, iY4.x);
        ftype4 tY2 = ftype4(iY1.y, iY2.y, iY3.y, iY4.y);
        ftype4 tY3 = ftype4(iY1.z, iY2.z, iY3.z, iY4.z);
        ftype4 tY4 = ftype4(iY1.w, iY2.w, iY3.w, iY4.w);
        r.x += dot(iX, tY1);
        r.y += dot(iX, tY2);
        r.z += dot(iX, tY3);
        r.w += dot(iX, tY4);
    }
    r += biasY.sample(sample, float2(0, 0), gid.z);
    r = activation(r, param);
    outTexture.write(r, gid.xy, gid.z);
}


kernel void mat_mul_2dims(texture2d_array<ftype, access::read> inputX[[texture(0)]],
    texture2d_array<ftype, access::read> inputY[[texture(1)]],
    texture2d_array<ftype, access::write> outTexture[[texture(2)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    ftype4 r = 0;
    int xLen = inputX.get_array_size();
    for (int i = 0; i < xLen; i++) {
        ftype4 iX = inputX.read(uint2(gid.x, gid.y), i);

        ftype4 iY1 = inputY.read(uint2(i * 4, gid.y), gid.z);
        ftype4 iY2 = inputY.read(uint2(i * 4 + 1, gid.y), gid.z);
        ftype4 iY3 = inputY.read(uint2(i * 4 + 2, gid.y), gid.z);
        ftype4 iY4 = inputY.read(uint2(i * 4 + 3, gid.y), gid.z);

        r.x += dot(iX, ftype4(iY1.x, iY2.x, iY3.x, iY4.x));
        r.y += dot(iX, ftype4(iY1.y, iY2.y, iY3.y, iY4.y));
        r.z += dot(iX, ftype4(iY1.z, iY2.z, iY3.z, iY4.z));
        r.w += dot(iX, ftype4(iY1.w, iY2.w, iY3.w, iY4.w));
    }
    outTexture.write(r, gid.xy, gid.z);
}


kernel void mat_mul_2dims_trans_y(texture2d_array<ftype, access::read> inputX[[texture(0)]],
    texture2d_array<ftype, access::read> inputY[[texture(1)]],
    texture2d_array<ftype, access::write> outTexture[[texture(2)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    ftype4 r = 0;
    int xLen = inputX.get_array_size();
    for (int i = 0; i < xLen; i++) {
        ftype4 iX = inputX.read(uint2(gid.x, gid.y), i);

        ftype4 iY1 = inputY.read(uint2(4 * gid.z, gid.y), i);
        ftype4 iY2 = inputY.read(uint2(4 * gid.z + 1, gid.y), i);
        ftype4 iY3 = inputY.read(uint2(4 * gid.z + 2, gid.y), i);
        ftype4 iY4 = inputY.read(uint2(4 * gid.z + 3, gid.y), i);

        r.x += dot(iX, iY1);
        r.y += dot(iX, iY2);
        r.z += dot(iX, iY3);
        r.w += dot(iX, iY4);
    }
    outTexture.write(r, gid.xy, gid.z);
}

kernel void mat_mul_4dims(texture2d_array<ftype, access::read> inputX[[texture(0)]],
    texture2d_array<ftype, access::read> inputY[[texture(1)]],
    texture2d_array<ftype, access::write> outTexture[[texture(2)]],
    constant MatmulParam& param[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    ftype4 r = 0;
    bool broadcast = param.broadcast;
    int gidz = broadcast ? 0 : gid.z;
    int xLen = inputX.get_width();
    for (int i = 0; i < xLen; i++) {
        ftype4 iX = inputX.read(uint2(i, gid.y), gid.z);
        ftype4 iY = inputY.read(uint2(gid.x, i), gidz);
        iY = broadcast ? ftype4(iY.x, iY.x, iY.x, iY.x) : iY;
        r += iX * iY;
    }
    outTexture.write(r, gid.xy, gid.z);
}

kernel void mat_mul_4dim_trans_y(texture2d_array<ftype, access::read> inputX[[texture(0)]],
    texture2d_array<ftype, access::read> inputY[[texture(1)]],
    texture2d_array<ftype, access::write> outTexture[[texture(2)]],
    constant MatmulParam& param[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    ftype4 r = 0;
    bool broadcast = param.broadcast;
    int gidz = broadcast ? 0 : gid.z;
    int xLen = inputX.get_width();
    for (int i = 0; i < xLen; i++) {
        ftype4 iX = inputX.read(uint2(i, gid.y), gid.z);
        ftype4 iY = inputY.read(uint2(i, gid.x), gidz);
        iY = broadcast ? ftype4(iY.x, iY.x, iY.x, iY.x) : iY;
        r += iX * iY;
    }
    outTexture.write(r, gid.xy, gid.z);
}

kernel void mat_mul_4dim_trans_x(texture2d_array<ftype, access::read> inputX[[texture(0)]],
    texture2d_array<ftype, access::read> inputY[[texture(1)]],
    texture2d_array<ftype, access::write> outTexture[[texture(2)]],
    constant MatmulParam& param[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    ftype4 r = 0;
    bool broadcast = param.broadcast;
    int gidz = broadcast ? 0 : gid.z;
    int xLen = inputX.get_height();
    for (int i = 0; i < xLen; i++) {
        ftype4 iX = inputX.read(uint2(gid.y, i), gid.z);
        ftype4 iY = inputY.read(uint2(gid.x, i), gidz);
        iY = broadcast ? ftype4(iY.x, iY.x, iY.x, iY.x) : iY;
        r += iX * iY;
    }
    outTexture.write(r, gid.xy, gid.z);
}

kernel void mat_mul_4dim_trans_xy(texture2d_array<ftype, access::read> inputX[[texture(0)]],
    texture2d_array<ftype, access::read> inputY[[texture(1)]],
    texture2d_array<ftype, access::write> outTexture[[texture(2)]],
    constant MatmulParam& param[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    ftype4 r = 0;
    bool broadcast = param.broadcast;
    int gidz = broadcast ? 0 : gid.z;
    int xLen = inputX.get_height();
    for (int i = 0; i < xLen; i++) {
        ftype4 iX = inputX.read(uint2(gid.y, i), gid.z);
        ftype4 iY = inputY.read(uint2(i, gid.x), gidz);
        iY = broadcast ? ftype4(iY.x, iY.x, iY.x, iY.x) : iY;
        r += iX * iY;
    }
    outTexture.write(r, gid.xy, gid.z);
}