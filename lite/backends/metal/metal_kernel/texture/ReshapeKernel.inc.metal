/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 
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

#define CONCAT2(a, b) a ## b
#define CONCAT2_(a, b) a ## _ ## b
#define CONCAT3_(a, b, c) a ## _ ## b ## _ ## c
#define CONCAT4_(a, b, c, d) a ## _ ## b ## _ ## c ## _ ## d

#define FUNC(f, r1, r2, p) CONCAT4_(f, r1, r2, p)
#define VECTOR(p, n) CONCAT2(p, n)
#define FUNC_R(f, r) CONCAT2_(f, r)

kernel void FUNC(reshape, RIN, ROUT, P)(texture2d_array<P, access::read> inTexture [[texture(0)]],
                                        texture2d_array<P, access::write> outTexture [[texture(1)]],
                                        constant ReshapeParam &rp [[buffer(0)]],
                                        uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) return;
    
    int oxyzn[4] = {int(gid.x), int(gid.y), int(gid.z), 0}, oabcd[4], ixyzn[4], iabcd[4];
    ReshapeParam lrp = rp;
    int oC = lrp.odim[lrp.otrans[3]];
    int iC = lrp.idim[lrp.itrans[3]];
    int count = lrp.odim[0] * lrp.odim[1] * lrp.odim[2] * lrp.odim[3];
    VECTOR(P, 4) r;
    for (int n = 0; n < 4; n++) {
        oxyzn[3] = n;
#if ROUT == 4
        xyzn2abcd_4(oC, oxyzn, oabcd);
#else
        FUNC_R(xyzn2abcd, ROUT)(oxyzn, oabcd);
#endif
        int tabcd[4];
        invtrans(lrp.otrans, oabcd, tabcd);
        int index = abcd2index(lrp.odim, tabcd);
        if (index < count) {
            index2abcd(lrp.idim, index, tabcd);
            trans(lrp.itrans, tabcd, iabcd);
#if RIN == 4
            abcd2xyzn_4(iC, iabcd, ixyzn);
#else
            FUNC_R(abcd2xyzn, RIN)(iabcd, ixyzn);
#endif
            r[n] = inTexture.read(uint2(ixyzn[0], ixyzn[1]), ixyzn[2])[ixyzn[3]];
        } else {
            r[n] = 0;
        }
    }
    outTexture.write(r, gid.xy, gid.z);
}

#endif
