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

kernel void split(texture2d_array<ftype, access::read> input[[texture(0)]],
    texture2d_array<ftype, access::write> out1[[texture(1)]],
    texture2d_array<ftype, access::write> out2[[texture(2)]],
    texture2d_array<ftype, access::write> out3[[texture(3)]],
    texture2d_array<ftype, access::write> out4[[texture(4)]],
    constant SplitParam& sp[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    int n = sp.num;
    int v_ = sp.v_;
    ftype4 r = input.read(gid.xy, gid.z);
    if (v_ == 1) {
        int x = gid.x - sp.offset;
        ;
        if (x < sp.vdim[0]) {
            out1.write(r, gid.xy, gid.z);
            return;
        }
        x -= sp.vdim[0];
        if (x < sp.vdim[1]) {
            out2.write(r, uint2(x, gid.y), gid.z);
            return;
        }
        if (n >= 3) {
            x -= sp.vdim[1];
            if (x < sp.vdim[2]) {
                out3.write(r, uint2(x, gid.y), gid.z);
                return;
            }
        }
        if (n >= 4) {
            x -= sp.vdim[2];
            if (x < sp.vdim[3]) {
                out4.write(r, uint2(x, gid.y), gid.z);
                return;
            }
        }
    } else if (v_ == 2) {
        int y = gid.y - sp.offset;
        if (y < sp.vdim[0]) {
            out1.write(r, gid.xy, gid.z);
            return;
        }
        y -= sp.vdim[0];
        if (y < sp.vdim[1]) {
            out2.write(r, uint2(gid.x, y), gid.z);
            return;
        }
        if (n >= 3) {
            y -= sp.vdim[1];
            if (y < sp.vdim[2]) {
                out3.write(r, uint2(gid.x, y), gid.z);
                return;
            }
        }
        if (n >= 4) {
            y -= sp.vdim[2];
            if (y < sp.vdim[3]) {
                out4.write(r, uint2(gid.x, y), gid.z);
                return;
            }
        }
    } else if (v_ == 3) {
        int z = gid.z;
        if (z < sp.vdim[0]) {
            out1.write(r, gid.xy, z);
            return;
        }
        z -= sp.vdim[0];
        if (z < sp.vdim[1]) {
            out2.write(r, gid.xy, z);
            return;
        }
        if (n >= 3) {
            z -= sp.vdim[1];
            if (z < sp.vdim[2]) {
                out3.write(r, gid.xy, z);
                return;
            }
        }
        if (n >= 4) {
            z -= sp.vdim[2];
            if (z < sp.vdim[3]) {
                out4.write(r, gid.xy, z);
                return;
            }
        }
    }
}


kernel void split_zz(texture2d_array<ftype, access::read> input[[texture(0)]],
    texture2d_array<ftype, access::write> out1[[texture(1)]],
    texture2d_array<ftype, access::write> out2[[texture(2)]],
    texture2d_array<ftype, access::write> out3[[texture(3)]],
    texture2d_array<ftype, access::write> out4[[texture(4)]],
    constant SplitParam& sp[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    int n = sp.num;
    int v_ = sp.v_;
    if (v_ == 4) {
        int z = gid.z;
        if (z - (sp.vdim[0] + 3) / 4 < 0) {  // output1
            ftype4 r = input.read(gid.xy, z);
            int len = (gid.z + 1) * 4 - sp.vdim[0];
            for (int i = 0; i < len; i++) {
                r[3 - i] = 0;
            }
            out1.write(r, gid.xy, gid.z);
            return;
        }
        z -= (sp.vdim[0] + 3) / 4;
        if (z - (sp.vdim[1] + 3) / 4 < 0) {
            int z_origin = z * 4 + sp.vdim[0];
            int z_end = min(z_origin + 3, sp.vdim[0] + sp.vdim[1] - 1);
            ftype4 r = 0;
            ftype4 r1 = input.read(gid.xy, z_origin / 4);
            int start = z_origin % 4;
            for (int i = start; i < 4 && i - start <= z_end - z_origin; i++) {
                r[i - start] = r1[i];
            }
            r1 = input.read(gid.xy, z_end / 4);
            int end = z_end % 4;
            for (int i = end; i >= 0 && end - i <= z_end - z_origin; i--) {
                r[z_end - z_origin + i - end] = r1[i];
            }
            out2.write(r, gid.xy, z);
            return;
        }
        if (n >= 3) {
            z -= (sp.vdim[1] + 3) / 4;
            if (z - (sp.vdim[2] + 3) / 4 < 0) {
                int z_origin = z * 4 + sp.vdim[0] + sp.vdim[1];
                int z_end = min(z_origin + 3, sp.vdim[0] + sp.vdim[1] + sp.vdim[2] - 1);
                ftype4 r = 0;
                ftype4 r1 = input.read(gid.xy, z_origin / 4);
                int start = z_origin % 4;
                for (int i = start; i < 4 && i - start <= z_end - z_origin; i++) {
                    r[i - start] = r1[i];
                }
                r1 = input.read(gid.xy, z_end / 4);
                int end = z_end % 4;
                for (int i = end; i >= 0 && end - i <= z_end - z_origin; i--) {
                    r[z_end - z_origin + i - end] = r1[i];
                }
                out3.write(r, gid.xy, z);
                return;
            }
        }
        if (n >= 4) {
            z -= (sp.vdim[2] + 2) / 4;
            if (z - (sp.vdim[3] + 2) / 4 < 0) {
                int z_origin = z * 4 + sp.vdim[0] + sp.vdim[1] + sp.vdim[2];
                int z_end =
                    min(z_origin + 3, sp.vdim[0] + sp.vdim[1] + sp.vdim[2] + sp.vdim[3] - 1);
                ftype4 r = 0;
                ftype4 r1 = input.read(gid.xy, z_origin / 4);
                int start = z_origin % 4;
                for (int i = start; i < 4 && i - start <= z_end - z_origin; i++) {
                    r[i - start] = r1[i];
                }
                r1 = input.read(gid.xy, z_end / 4);
                int end = z_end % 4;
                for (int i = end; i >= 0 && end - i <= z_end - z_origin; i--) {
                    r[z_end - z_origin + i - end] = r1[i];
                }
                out4.write(r, gid.xy, z);
                return;
            }
        }
    }
}
