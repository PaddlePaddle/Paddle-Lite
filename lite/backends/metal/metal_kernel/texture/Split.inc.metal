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

#define CONCAT2(a, b) a##b
#define CONCAT2_(a, b) a##_##b
#define CONCAT3_(a, b, c) a##_##b##_##c
#define CONCAT4_(a, b, c, d) a##_##b##_##c##_##d
#define CONCAT5_(a, b, c, d, e) a##_##b##_##c##_##d##_##e

#define FUNC(f, r, n, v) CONCAT4_(f, r, n, v)
#define VECTOR(p, n) CONCAT2(p, n)
#define FUNC_R(f, r) CONCAT2_(f, r)

#if V == VX
#define VV x
#elif V == VY
#define VV y
#elif V == VZ
#define VV z
#elif V == VZZ
#define VV zz
#else
#define VV normal
#endif

#if V == VY
kernel void FUNC(split, R, N, VV)(
    texture2d_array<P, access::read> input [[texture(0)]],
    texture2d_array<P, access::write> out1 [[texture(1)]],
    texture2d_array<P, access::write> out2 [[texture(2)]],
#if N >= 3
    texture2d_array<P, access::write> out3 [[texture(3)]],
#endif  // N >= 3
#if N >= 4
    texture2d_array<P, access::write> out4 [[texture(4)]],
#endif  // N >= 4
    constant SplitParam &sp [[buffer(0)]],
    uint3 gid [[thread_position_in_grid]]) {

  VECTOR(P, 4) r = input.read(gid.xy, gid.z);
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
#if N >= 3
  y -= sp.vdim[1];
  if (y < sp.vdim[2]) {
    out3.write(r, uint2(gid.x, y), gid.z);
    return;
  }
#endif  // N >= 3
#if N >= 4
  y -= sp.vdim[2];
  if (y < sp.vdim[3]) {
    out4.write(r, uint2(gid.x, y), gid.z);
    return;
  }
#endif  // N >= 4
}
#endif  // V == VY

#if V == VX
kernel void FUNC(split, R, N, VV)(
    texture2d_array<P, access::read> input [[texture(0)]],
    texture2d_array<P, access::write> out1 [[texture(1)]],
    texture2d_array<P, access::write> out2 [[texture(2)]],
#if N >= 3
    texture2d_array<P, access::write> out3 [[texture(3)]],
#endif  // N >= 3
#if N >= 4
    texture2d_array<P, access::write> out4 [[texture(4)]],
#endif  // N >= 4
    constant SplitParam &sp [[buffer(0)]],
    uint3 gid [[thread_position_in_grid]]) {
  VECTOR(P, 4) r = input.read(gid.xy, gid.z);
  int x = gid.x;
  if (x < sp.vdim[0]) {
    out1.write(r, gid.xy, gid.z);
    return;
  }
  x -= sp.vdim[0];
  if (x < sp.vdim[1]) {
    out2.write(r, uint2(x, gid.y), gid.z);
    return;
  }
#if N >= 3
  x -= sp.vdim[1];
  if (x < sp.vdim[2]) {
    out3.write(r, uint2(x, gid.y), gid.z);
    return;
  }
#endif  // N >= 3
#if N >= 4
  x -= sp.vdim[2];
  if (x < sp.vdim[3]) {
    out4.write(r, uint2(x, gid.y), gid.z);
    return;
  }
#endif  // N >= 4
}
#endif  // V == VX

#if V == VZ
kernel void FUNC(split, R, N, VV)(
    texture2d_array<P, access::read> input [[texture(0)]],
    texture2d_array<P, access::write> out1 [[texture(1)]],
    texture2d_array<P, access::write> out2 [[texture(2)]],
#if N >= 3
    texture2d_array<P, access::write> out3 [[texture(3)]],
#endif  // N >= 3
#if N >= 4
    texture2d_array<P, access::write> out4 [[texture(4)]],
#endif  // N >= 4
    constant SplitParam &sp [[buffer(0)]],
    uint3 gid [[thread_position_in_grid]]) {
  VECTOR(P, 4) r = input.read(gid.xy, gid.z);
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
#if N >= 3
  z -= sp.vdim[1];
  if (z < sp.vdim[2]) {
    out3.write(r, gid.xy, z);
    return;
  }
#endif  // N >= 3
#if N >= 4
  z -= sp.vdim[2];
  if (z < sp.vdim[3]) {
    out4.write(r, gid.xy, z);
    return;
  }
#endif  // N >= 4
}
#endif  // V == VZ

#if V == VZZ
kernel void FUNC(split, R, N, VV)(
    texture2d_array<P, access::read> input [[texture(0)]],
    texture2d_array<P, access::write> out1 [[texture(1)]],
    texture2d_array<P, access::write> out2 [[texture(2)]],
#if N >= 3
    texture2d_array<P, access::write> out3 [[texture(3)]],
#endif  // N >= 3
#if N >= 4
    texture2d_array<P, access::write> out4 [[texture(4)]],
#endif  // N >= 4
    constant SplitParam &sp [[buffer(0)]],
    uint3 gid [[thread_position_in_grid]]) {
  int index = 0;
  int z = gid.z;
  if (z - (sp.vdim[0] + 3) / 4 < 0) {  // output1
    VECTOR(P, 4) r = input.read(gid.xy, z);
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
    VECTOR(P, 4) r;
    r[0] = 0;
    r[1] = 0;
    r[2] = 0;
    r[3] = 0;
    VECTOR(P, 4) r1 = input.read(gid.xy, z_origin / 4);
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
#if N >= 3
  z -= (sp.vdim[1] + 3) / 4;
  if (z - (sp.vdim[2] + 3) / 4 < 0) {
    int z_origin = z * 4 + sp.vdim[0] + sp.vdim[1];
    int z_end = min(z_origin + 3, sp.vdim[0] + sp.vdim[1] + sp.vdim[2] - 1);
    VECTOR(P, 4) r;
    r[0] = 0;
    r[1] = 0;
    r[2] = 0;
    r[3] = 0;
    VECTOR(P, 4) r1 = input.read(gid.xy, z_origin / 4);
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
#endif  // N >= 3
#if N >= 4
  z -= (sp.vdim[2] + 2) / 4;
  if (z - (sp.vdim[3] + 2) / 4 < 0) {
    int z_origin = z * 4 + sp.vdim[0] + sp.vdim[1] + sp.vdim[2];
    int z_end = min(z_origin + 3,
                    sp.vdim[0] + sp.vdim[1] + sp.vdim[2] + sp.vdim[3] - 1);
    VECTOR(P, 4) r;
    r[0] = 0;
    r[1] = 0;
    r[2] = 0;
    r[3] = 0;
    VECTOR(P, 4) r1 = input.read(gid.xy, z_origin / 4);
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
#endif  // N >= 4
}
#endif  // V == VZZ

#undef VV
#endif
