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

kernel void reshape(texture2d_array<ftype, access::read> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    constant ReshapeParam& rp[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;

    // output coordinate: GPU coordinate is {x,y,z,n}
    // One location contains four data, every data is on the C channel
    int oxyzn[4] = {int(gid.x), int(gid.y), int(gid.z), 0};
    // output coordinate: CPU: {z/outC, y, x, z%outC} layout: NHWC
    // Tensor data: after data conversion, before uploading to the GPU
    int oabcd[4];
    // input coordinate: GPU
    int ixyzn[4];
    // input coordinate: CPU
    // Tensor data: after data conversion, before uploading to the GPU
    int iabcd[4];

    ftype4 r = ftype4(0.0);
    ReshapeParam lrp = rp;
    int count = lrp.odim[0] * lrp.odim[1] * lrp.odim[2] * lrp.odim[3];
    for (int n = 0; n < 4; n++) {
        oxyzn[3] = n;
        // size of C channnel
        int oC = lrp.odim[lrp.otrans[3]];
        // GPU coordinate to NCHW coordinate
        // That is, the position represented by the data before uploading to the GPU
        xyzn2abcd_4(oC, oxyzn, oabcd);
        int tabcd[4];
        // Convert coordinates according to Tensor conversion
        // attention: The same logic with the 'initTexture' logic in 'metal_image'
        // 4-dims conversion: Tensor NCHW->NHWC  3-dims isn't converted
        // eg1: tensor={1, 24, 208, 208} -> dim={1, 208, 208, 24}
        // eg2: tensor={1, 9, 3549} -> dim={1, 1, 9, 3549}
        invtrans(lrp.otrans, oabcd, tabcd);
        // CPU NHWC coordinate -> CPU NHWC coordinate
        int index = abcd2index(lrp.odim, tabcd);
        if (index < count) {
            // The following logic is consistent with the above, just the opposite process
            index2abcd(lrp.idim, index, tabcd);
            trans(lrp.itrans, tabcd, iabcd);
            int iC = lrp.idim[lrp.itrans[3]];
            abcd2xyzn_4(iC, iabcd, ixyzn);
            r[n] = inTexture.read(uint2(ixyzn[0], ixyzn[1]), ixyzn[2])[ixyzn[3]];
        } else {
            r[n] = 0;
        }
    }
    outTexture.write(r, gid.xy, gid.z);
}
