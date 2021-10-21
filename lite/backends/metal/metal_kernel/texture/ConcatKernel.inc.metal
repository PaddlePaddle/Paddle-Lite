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

#define CONCAT2(a, b) a##b
#define CONCAT4_(a, b, c, d) a##_##b##_##c##_##d

#define FUNC(f, r, n, v) CONCAT4_(f, r, n, v)
#define VECTOR(p, n) CONCAT2(p, n)

#if V == VNORMAL
kernel void FUNC(concat, R, N, normal)(texture2d_array<P, access::read> in0[[texture(0)]],
    texture2d_array<P, access::read> in1[[texture(1)]],
#if N >= 3
    texture2d_array<P, access::read> in2[[texture(2)]],
#endif
#if N >= 4
    texture2d_array<P, access::read> in3[[texture(3)]],
#endif
#if N >= 5
    texture2d_array<P, access::read> in4[[texture(4)]],
#endif
#if N >= 6
    texture2d_array<P, access::read> in5[[texture(5)]],
#endif
    texture2d_array<P, access::read> inx[[texture(N)]],
    texture2d_array<P, access::write> out[[texture(N + 1)]],
    constant ConcatParam& pm[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {

    ConcatParam cp = pm;
    int xyzn[4] = {int(gid.x), int(gid.y), int(gid.z), 0}, abcd[4], oxyzn[4];
    VECTOR(P, 4) r = inx.read(gid.xy, gid.z);
    for (int i = 0; i < 4; i++) {
        xyzn[3] = i;
        xyzn2abcd_4(cp.odim[3], xyzn, abcd);
        int k = abcd[cp.axis] - cp.offset;
        if (k < 0) continue;
        int j = 0;
        for (; j < N; j++) {
            if (k < cp.vdim[j]) {
                break;
            }
            k -= cp.vdim[j];
        }
        if (j == N) {
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
#if N >= 3
            case 2:
                r[i] = in2.read(uint2(oxyzn[0], oxyzn[1]), oxyzn[2])[oxyzn[3]];
                break;
#endif
#if N >= 4
            case 3:
                r[i] = in3.read(uint2(oxyzn[0], oxyzn[1]), oxyzn[2])[oxyzn[3]];
                break;
#endif
#if N >= 5
            case 4:
                r[i] = in4.read(uint2(oxyzn[0], oxyzn[1]), oxyzn[2])[oxyzn[3]];
                break;
#endif
#if N >= 6
            case 5:
                r[i] = in5.read(uint2(oxyzn[0], oxyzn[1]), oxyzn[2])[oxyzn[3]];
                break;
#endif
        }
    }
    out.write(r, gid.xy, gid.z);
}

#endif  // V == NORMAL

#if V == VX
kernel void FUNC(concat, R, N, x)(texture2d_array<P, access::read> in0[[texture(0)]],
    texture2d_array<P, access::read> in1[[texture(1)]],
#if N >= 3
    texture2d_array<P, access::read> in2[[texture(2)]],
#endif  // N >= 3
#if N >= 4
    texture2d_array<P, access::read> in3[[texture(3)]],
#endif  // N >= 4
#if N >= 5
    texture2d_array<P, access::read> in4[[texture(4)]],
#endif  // N >= 5
#if N >= 6
    texture2d_array<P, access::read> in5[[texture(5)]],
#endif  // N >= 6
    texture2d_array<P, access::write> out[[texture(N)]],
    constant ConcatParam& pm[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    int x = gid.x - pm.offset;
    if (x < 0) return;
    if (x < pm.vdim[0]) {
        VECTOR(P, 4) r = in0.read(gid.xy, gid.z);
        out.write(r, gid.xy, gid.z);
        return;
    }
    x -= pm.vdim[0];
    if (x < pm.vdim[1]) {
        VECTOR(P, 4) r = in1.read(uint2(x, gid.y), gid.z);
        out.write(r, gid.xy, gid.z);
        return;
    }
#if N >= 3
    x -= pm.vdim[1];
    if (x < pm.vdim[2]) {
        VECTOR(P, 4) r = in2.read(uint2(x, gid.y), gid.z);
        out.write(r, gid.xy, gid.z);
        return;
    }
#endif  // N >= 3
#if N >= 4
    x -= pm.vdim[2];
    if (x < pm.vdim[3]) {
        VECTOR(P, 4) r = in3.read(uint2(x, gid.y), gid.z);
        out.write(r, gid.xy, gid.z);
        return;
    }
#endif  // N >= 4
#if N >= 5
    x -= pm.vdim[3];
    if (x < pm.vdim[4]) {
        VECTOR(P, 4) r = in4.read(uint2(x, gid.y), gid.z);
        out.write(r, gid.xy, gid.z);
        return;
    }
#endif  // N >= 5
#if N >= 6
    x -= pm.vdim[4];
    if (x < pm.vdim[5]) {
        VECTOR(P, 4) r = in5.read(uint2(x, gid.y), gid.z);
        out.write(r, gid.xy, gid.z);
        return;
    }
#endif  // N >= 6
}
#endif  // V == VX

#if V == VY
kernel void FUNC(concat, R, N, y)(texture2d_array<P, access::read> in0[[texture(0)]],
    texture2d_array<P, access::read> in1[[texture(1)]],
#if N >= 3
    texture2d_array<P, access::read> in2[[texture(2)]],
#endif  // N >= 3
#if N >= 4
    texture2d_array<P, access::read> in3[[texture(3)]],
#endif  // N >= 4
#if N >= 5
    texture2d_array<P, access::read> in4[[texture(4)]],
#endif  // N >= 5
#if N >= 6
    texture2d_array<P, access::read> in5[[texture(5)]],
#endif  // N >= 6
    texture2d_array<P, access::write> out[[texture(N)]],
    constant ConcatParam& pm[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    int y = gid.y - pm.offset;
    if (y < 0) return;
    if (y < pm.vdim[0]) {
        VECTOR(P, 4) r = in0.read(gid.xy, gid.z);
        out.write(r, gid.xy, gid.z);
        return;
    }
    y -= pm.vdim[0];
    if (y < pm.vdim[1]) {
        VECTOR(P, 4) r = in1.read(uint2(gid.x, y), gid.z);
        out.write(r, gid.xy, gid.z);
        return;
    }
#if N >= 3
    y -= pm.vdim[1];
    if (y < pm.vdim[2]) {
        VECTOR(P, 4) r = in2.read(uint2(gid.x, y), gid.z);
        out.write(r, gid.xy, gid.z);
        return;
    }
#endif  // N >= 3
#if N >= 4
    y -= pm.vdim[2];
    if (y < pm.vdim[3]) {
        VECTOR(P, 4) r = in3.read(uint2(gid.x, y), gid.z);
        out.write(r, gid.xy, gid.z);
        return;
    }
#endif  // N >= 4
#if N >= 5
    y -= pm.vdim[3];
    if (y < pm.vdim[4]) {
        VECTOR(P, 4) r = in4.read(uint2(gid.x, y), gid.z);
        out.write(r, gid.xy, gid.z);
        return;
    }
#endif  // N >= 5
#if N >= 6
    y -= pm.vdim[4];
    if (y < pm.vdim[5]) {
        VECTOR(P, 4) r = in5.read(uint2(gid.x, y), gid.z);
        out.write(r, gid.xy, gid.z);
        return;
    }
#endif  // N >= 6
}
#endif  // V == VY

#if V == VZ
kernel void FUNC(concat, R, N, z)(texture2d_array<P, access::read> in0[[texture(0)]],
    texture2d_array<P, access::read> in1[[texture(1)]],
#if N >= 3
    texture2d_array<P, access::read> in2[[texture(2)]],
#endif  // N >= 3
#if N >= 4
    texture2d_array<P, access::read> in3[[texture(3)]],
#endif  // N >= 4
#if N >= 5
    texture2d_array<P, access::read> in4[[texture(4)]],
#endif  // N >= 5
#if N >= 6
    texture2d_array<P, access::read> in5[[texture(5)]],
#endif  // N >= 6
    texture2d_array<P, access::write> out[[texture(N)]],
    constant ConcatParam& pm[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    int z = gid.z - pm.offset;
    if (z < 0) return;
    if (z < pm.vdim[0]) {
        VECTOR(P, 4) r = in0.read(gid.xy, gid.z);
        out.write(r, gid.xy, gid.z);
        return;
    }
    z -= pm.vdim[0];
    if (z < pm.vdim[1]) {
        VECTOR(P, 4) r = in1.read(gid.xy, z);
        out.write(r, gid.xy, gid.z);
        return;
    }
#if N >= 3
    z -= pm.vdim[1];
    if (z < pm.vdim[2]) {
        VECTOR(P, 4) r = in2.read(gid.xy, z);
        out.write(r, gid.xy, gid.z);
        return;
    }
#endif  // N >= 3
#if N >= 4
    z -= pm.vdim[2];
    if (z < pm.vdim[3]) {
        VECTOR(P, 4) r = in3.read(gid.xy, z);
        out.write(r, gid.xy, gid.z);
        return;
    }
#endif  // N >= 4
#if N >= 5
    z -= pm.vdim[3];
    if (z < pm.vdim[4]) {
        VECTOR(P, 4) r = in4.read(gid.xy, z);
        out.write(r, gid.xy, gid.z);
        return;
    }
#endif  // N >= 5
#if N >= 6
    z -= pm.vdim[4];
    if (z < pm.vdim[5]) {
        VECTOR(P, 4) r = in5.read(gid.xy, z);
        out.write(r, gid.xy, gid.z);
        return;
    }
#endif  // N >= 6
}
#endif  // V == VZ

#endif  // #ifdef P
