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

struct TransposeParam {
    int idim[4];    // input dim [NCHW-CPU]-> [NHWC-GPU]
    int itrans[4];  // input transpose dim
    int odim[4];    // output dim [NCHW-CPU]-> [NHWC-GPU]
    int otrans[4];  // output transpose dim
    int axis[4];
};

using namespace metal;

kernel void transpose(texture2d_array<ftype, access::read> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    constant TransposeParam& pm[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    ftype4 r;
    // output coordinate GPU {tex_w, tex_h, tex_arraylength, i}
    int x_xyzn[4] = {int32_t(gid.x), int32_t(gid.y), int32_t(gid.z), 0};
    // output coordinate CPU NHWC
    int x_abcd[4];
    // output coordinate after conversion CPU (eg:NHWC->NCHW)
    int tx_abcd[4];
    // output transpose dims
    int xtrans[4] = {pm.otrans[0], pm.otrans[1], pm.otrans[2], pm.otrans[3]};

    // output transpose dims
    int ytrans[4] = {pm.itrans[0], pm.itrans[1], pm.itrans[2], pm.itrans[3]};
    // output coordinate after conversion CPU (eg:NHWC->NCHW)
    int ty_abcd[4];
    // input coordinate CPU NHWC
    int32_t y_abcd[4] = {0, 0, 0, 0};
    // input coordinate GPU  {tex_w, tex_h, tex_arraylength, i}
    int32_t y_xyzn[4];

    for (int n = 0; n < 4; n++) {
        // output Index值
        x_xyzn[3] = n;
        // output size of C channnel
        int oC = pm.odim[pm.otrans[3]];
        // output [WH(NC)-GPU] -> [NHWC-CPU]
        xyzn2abcd(oC, x_xyzn, x_abcd);
        // output [NHWC-CPU] -> [NCHW-CPU]
        invtrans(xtrans, x_abcd, tx_abcd);
        // input align output
        ty_abcd[pm.axis[0]] = tx_abcd[0];
        ty_abcd[pm.axis[1]] = tx_abcd[1];
        ty_abcd[pm.axis[2]] = tx_abcd[2];
        ty_abcd[pm.axis[3]] = tx_abcd[3];
        // input [NCHW-CPU] -> [NHWC-CPU]
        trans(ytrans, ty_abcd, y_abcd);
        // input size of C channnel
        int iC = pm.idim[pm.itrans[3]];
        // input [NHWC-CPU] -> [WHNC-GPU]
        abcd2xyzn(iC, y_abcd, y_xyzn);
        // read Y
        r[n] = inTexture.read(uint2(y_xyzn[0], y_xyzn[1]), y_xyzn[2])[y_xyzn[3]];
    }
    outTexture.write(r, gid.xy, gid.z);
}
