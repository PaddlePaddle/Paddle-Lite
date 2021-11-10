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

kernel void elementwise(texture2d_array<ftype, access::read> inputX[[texture(0)]],
    texture2d_array<ftype, access::read> inputY[[texture(1)]],
    texture2d_array<ftype, access::write> outTexture[[texture(2)]],
    constant ElementwiseAddParam& pm[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    ftype4 rx, ry;
    rx = inputX.read(gid.xy, gid.z);
    // elementwise
    if (pm.fast == 1) {
        ry = inputY.read(gid.xy, gid.z);
    }
    // add at one number
    else if (pm.ByNum == 1) {
        ry = inputY.read(uint2(0, 0), gid.z);
    }
    // add at C channel
    else if (pm.addByChannel == 1) {
        ry = inputY.read(uint2(0, 0), gid.z);
    }
    // add at HW
    else if (pm.ByHW == 1) {
        ry = inputY.read(gid.xy, 0);
    }
    // add at W
    else if (pm.ByW == 1) {
        ry = inputY.read(uint2(gid.x, 0), 0);
    } else {
        // X coordinate GPU NHNC
        int32_t x_xyzn[4] = {int32_t(gid.x), int32_t(gid.y), int32_t(gid.z), 0};
        // X coordinate CPU NHWC
        int32_t x_abcd[4];
        // X coordinate after conversion CPU (eg:NHWC->NCHW) 注意：有Y复用了这个值
        // attention: Y coordinate also use this value
        int32_t t_abcd[4];
        // Y coordinate CPU NHWC
        int32_t y_abcd[4] = {0, 0, 0, 0};
        // Y coordinate GPU NHNC
        int32_t y_xyzn[4];
        // X transpose dims
        int32_t xtrans[4] = {pm.xtrans[0], pm.xtrans[1], pm.xtrans[2], pm.xtrans[3]};
        // Y transpose dims
        int32_t ytrans[4] = {pm.ytrans[0], pm.ytrans[1], pm.ytrans[2], pm.ytrans[3]};
        // attribute
        int32_t yshift = 4 - pm.ylen - pm.axis;
        // use X coordinate to calculate Y coordinate , then read Y data
        for (int n = 0; n < 4; n++) {
            // ry Index值
            x_xyzn[3] = n;
            // X [WHNC-GPU] -> [NHWC-CPU]
            xyzn2abcd(pm.xdim[3], x_xyzn, x_abcd);
            // X [NHWC-CPU] -> [NCHW-CPU]
            invtrans(xtrans, x_abcd, t_abcd);
            // Y align X [NCHW-CPU]
            for (int k = pm.axis; k < (pm.axis + pm.ylen); k++) {
                y_abcd[yshift + k] = t_abcd[k];
            }
            // Y [NCHW-CPU] -> [NHWC-CPU]
            trans(ytrans, y_abcd, t_abcd);
            // Y [NHWC-CPU] -> [WHNC-GPU]
            abcd2xyzn(pm.ydim[3], t_abcd, y_xyzn);
            // read Y
            ry[n] = inputY.read(uint2(y_xyzn[0], y_xyzn[1]), y_xyzn[2])[y_xyzn[3]];
        }
    }
    ftype4 r;
    if (pm.arithmetic_type == 0)
        r = rx + ry;
    else if (pm.arithmetic_type == 1)
        r = rx - ry;
    else if (pm.arithmetic_type == 2)
        r = rx * ry;
    else if (pm.arithmetic_type == 3)
        r = rx / ry;
    outTexture.write(r, gid.xy, gid.z);
}

kernel void elementwise_relu(texture2d_array<ftype, access::read> inputX[[texture(0)]],
    texture2d_array<ftype, access::read> inputY[[texture(1)]],
    texture2d_array<ftype, access::write> outTexture[[texture(2)]],
    constant ElementwiseAddParam& pm[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    ftype4 rx, ry;
    rx = inputX.read(gid.xy, gid.z);
    if (pm.fast == 1) {
        ry = inputY.read(gid.xy, gid.z);
    } else if (pm.addByChannel == 1) {
        ry = inputY.read(uint2(0, 0), gid.z);
    } else {
        int32_t x_xyzn[4] = {int32_t(gid.x), int32_t(gid.y), int32_t(gid.z), 0}, x_abcd[4],
                t_abcd[4];
        int32_t y_abcd[4] = {0, 0, 0, 0}, y_xyzn[4];
        int32_t xtrans[4] = {pm.xtrans[0], pm.xtrans[1], pm.xtrans[2], pm.xtrans[3]};
        int32_t ytrans[4] = {pm.ytrans[0], pm.ytrans[1], pm.ytrans[2], pm.ytrans[3]};
        int32_t yshift = 4 - pm.ylen - pm.axis;
        for (int n = 0; n < 4; n++) {
            x_xyzn[3] = n;
            xyzn2abcd(pm.xdim[3], x_xyzn, x_abcd);
            invtrans(xtrans, x_abcd, t_abcd);
            for (int k = pm.axis; k < (pm.axis + pm.ylen); k++) {
                y_abcd[yshift + k] = t_abcd[k];
            }
            trans(ytrans, y_abcd, t_abcd);
            abcd2xyzn(pm.ydim[3], t_abcd, y_xyzn);
            ry[n] = inputY.read(uint2(y_xyzn[0], y_xyzn[1]), y_xyzn[2])[y_xyzn[3]];
        }
    }
    ftype4 output = rx + ry;

    // relu
    output = fmax(output, 0.0);

    outTexture.write(output, gid.xy, gid.z);
}