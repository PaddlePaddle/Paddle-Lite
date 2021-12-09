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

struct ConcatParam {
    int32_t odim[4];
    int32_t axis;
    int32_t offset;
    int32_t num;
    int32_t v_;
    int32_t trans[4];
    int32_t vdim[6];
};

kernel void concat_normal(texture2d_array<ftype, access::read> inx[[texture(0)]],
    texture2d_array<ftype, access::write> out[[texture(1)]],
    texture2d_array<ftype, access::read> in0[[texture(2)]],
    texture2d_array<ftype, access::read> in1[[texture(3)]],
    texture2d_array<ftype, access::read> in2[[texture(4)]],
    texture2d_array<ftype, access::read> in3[[texture(5)]],
    texture2d_array<ftype, access::read> in4[[texture(6)]],
    texture2d_array<ftype, access::read> in5[[texture(7)]],
    constant ConcatParam& pm[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    ConcatParam cp = pm;
    int n = pm.num;
    int xyzn[4] = {int(gid.x), int(gid.y), int(gid.z), 0}, abcd[4], oxyzn[4];
    ftype4 r = inx.read(gid.xy, gid.z);
    for (int i = 0; i < 4; i++) {
        xyzn[3] = i;
        xyzn2abcd_4(cp.odim[3], xyzn, abcd);
        int k = abcd[cp.axis] - cp.offset;
        if (k < 0) continue;
        int j = 0;
        for (; j < n; j++) {
            if (k < cp.vdim[j]) {
                break;
            }
            k -= cp.vdim[j];
        }
        if (j == n) {
            continue;
        }
        int ta = cp.odim[cp.axis];
        abcd[cp.axis] = k;
        cp.odim[cp.axis] = cp.vdim[j];
        abcd2xyzn_4(cp.odim[3], abcd, oxyzn);
        cp.odim[cp.axis] = ta;
        switch (j) {
            case 0:
                r[i] = in0.read(uint2(oxyzn[0], oxyzn[1]), oxyzn[2])[oxyzn[3]];
                break;
            case 1:
                r[i] = in1.read(uint2(oxyzn[0], oxyzn[1]), oxyzn[2])[oxyzn[3]];
                break;
            case 2:
                r[i] = in2.read(uint2(oxyzn[0], oxyzn[1]), oxyzn[2])[oxyzn[3]];
                break;
            case 3:
                r[i] = in3.read(uint2(oxyzn[0], oxyzn[1]), oxyzn[2])[oxyzn[3]];
                break;
            case 4:
                r[i] = in4.read(uint2(oxyzn[0], oxyzn[1]), oxyzn[2])[oxyzn[3]];
                break;
            case 5:
                r[i] = in5.read(uint2(oxyzn[0], oxyzn[1]), oxyzn[2])[oxyzn[3]];
                break;
        }
    }
    out.write(r, gid.xy, gid.z);
}

kernel void concat(texture2d_array<ftype, access::write> out[[texture(0)]],
    texture2d_array<ftype, access::read> in0[[texture(1)]],
    texture2d_array<ftype, access::read> in1[[texture(2)]],
    texture2d_array<ftype, access::read> in2[[texture(3)]],
    texture2d_array<ftype, access::read> in3[[texture(4)]],
    texture2d_array<ftype, access::read> in4[[texture(5)]],
    texture2d_array<ftype, access::read> in5[[texture(6)]],
    constant ConcatParam& pm[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    int n = pm.num;
    int v_ = pm.v_;
    if (v_ == 2) {
        int x = gid.x - pm.offset;
        if (x < 0) return;
        ftype4 r;
        for (int i = 0; i < n; i++) {
            if (i > 0) x -= pm.vdim[i - 1];
            if (x < pm.vdim[i]) {
                switch (i) {
                    case 0:
                        r = in0.read(gid.xy, gid.z);
                        break;
                    case 1:
                        r = in1.read(uint2(x, gid.y), gid.z);
                        break;
                    case 2:
                        r = in2.read(uint2(x, gid.y), gid.z);
                        break;
                    case 3:
                        r = in3.read(uint2(x, gid.y), gid.z);
                        break;
                    case 4:
                        r = in4.read(uint2(x, gid.y), gid.z);
                        break;
                    case 5:
                        r = in5.read(uint2(x, gid.y), gid.z);
                        break;
                }
                out.write(r, gid.xy, gid.z);
                return;
            }
        }
    } else if (v_ == 3) {
        int y = gid.y - pm.offset;
        if (y < 0) return;
        ftype4 r;
        for (int i = 0; i < n; i++) {
            if (i > 0) y -= pm.vdim[i - 1];
            if (y < pm.vdim[i]) {
                switch (i) {
                    case 0:
                        r = in0.read(gid.xy, gid.z);
                        break;
                    case 1:
                        r = in1.read(uint2(gid.x, y), gid.z);
                        break;
                    case 2:
                        r = in2.read(uint2(gid.x, y), gid.z);
                        break;
                    case 3:
                        r = in3.read(uint2(gid.x, y), gid.z);
                        break;
                    case 4:
                        r = in4.read(uint2(gid.x, y), gid.z);
                        break;
                    case 5:
                        r = in5.read(uint2(gid.x, y), gid.z);
                        break;
                }
                out.write(r, gid.xy, gid.z);
                return;
            }
        }
    } else if (v_ == 4) {
        int z = gid.z - pm.offset;
        if (z < 0) return;
        ftype4 r;
        for (int i = 0; i < n; i++) {
            if (i > 0) z -= pm.vdim[i - 1];
            if (z < pm.vdim[i]) {
                switch (i) {
                    case 0:
                        r = in0.read(gid.xy, gid.z);
                        break;
                    case 1:
                        r = in1.read(gid.xy, z);
                        break;
                    case 2:
                        r = in2.read(gid.xy, z);
                        break;
                    case 3:
                        r = in3.read(gid.xy, z);
                        break;
                    case 4:
                        r = in4.read(gid.xy, z);
                        break;
                    case 5:
                        r = in5.read(gid.xy, z);
                        break;
                }
                out.write(r, gid.xy, gid.z);
                return;
            }
        }
    }
}
