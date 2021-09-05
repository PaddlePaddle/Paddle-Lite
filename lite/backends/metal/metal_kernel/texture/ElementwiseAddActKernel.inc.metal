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

#ifdef P

#include <metal_stdlib>

#include "Macro.metal"

using namespace metal;

kernel void FUNC2_(elementwise_add, ACT_TYPE)(texture2d_array<P, access::read> inputX[[texture(0)]],
    texture2d_array<P, access::read> inputY[[texture(1)]],
    texture2d_array<P, access::write> outTexture[[texture(2)]],
    constant ElementwiseAddParam& pm[[buffer(0)]],
#ifdef PRELU_CHANNEL
    const device VECTOR(P, 4) * alpha[[buffer(1)]],
#endif
#ifdef PRELU_ELEMENT
    const device VECTOR(P, 4) * alpha[[buffer(1)]],
#endif
#ifdef PRELU_OTHER
    const device P* alpha[[buffer(1)]],
#endif
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    VECTOR(P, 4) rx, ry;
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
    VECTOR(P, 4) output = rx + ry;

#ifdef PRELU_CHANNEL
    VECTOR(P, 4) alpha_value = alpha[gid.z];
    output.x = output.x > 0 ? output.x : (alpha_value.x * output.x);
    output.y = output.y > 0 ? output.y : (alpha_value.y * output.y);
    output.z = output.z > 0 ? output.z : (alpha_value.z * output.z);
    output.w = output.w > 0 ? output.w : (alpha_value.w * output.w);
#endif
#ifdef PRELU_ELEMENT
    int alpha_to = (gid.y * outTexture.get_width() + gid.x) * outTexture.get_array_size();
    VECTOR(P, 4) alpha_value = alpha[alpha_to + gid.z];
    output.x = output.x > 0 ? output.x : (alpha_value.x * output.x);
    output.y = output.y > 0 ? output.y : (alpha_value.y * output.y);
    output.z = output.z > 0 ? output.z : (alpha_value.z * output.z);
    output.w = output.w > 0 ? output.w : (alpha_value.w * output.w);
#endif
#ifdef PRELU_OTHER
    P alpha_value = alpha[0];
    output.x = output.x > 0 ? output.x : (alpha_value * output.x);
    output.y = output.y > 0 ? output.y : (alpha_value * output.y);
    output.z = output.z > 0 ? output.z : (alpha_value * output.z);
    output.w = output.w > 0 ? output.w : (alpha_value * output.w);
#endif
#ifdef RELU
    output = fmax(output, 0.0);
#endif

    outTexture.write(output, gid.xy, gid.z);
}

#endif
