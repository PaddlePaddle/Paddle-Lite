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

#define FUNC(f, r, p) CONCAT3_(f, r, p)
#define VECTOR(p, n) CONCAT2(p, n)

kernel void FUNC(transpose, R, P)(texture2d_array<P, access::read> inTexture
                                  [[texture(0)]],
                                  texture2d_array<P, access::write> outTexture
                                  [[texture(1)]],
                                  constant TransposeParam &pm [[buffer(0)]],
                                  uint3 gid [[thread_position_in_grid]]) {
  VECTOR(P, 4) r;
  int oxyzn[4] = {int(gid.x), int(gid.y), int(gid.z), 0};
  int iabcd[4], oabcd[4], ixyzn[4];
  for (int n = 0; n < 4; n++) {
    oxyzn[3] = n;
#if R == 4
    xyzn2abcd_4(pm.oC, oxyzn, oabcd);
#endif  // R == 4
#if R == 3
    xyzn2abcd_3(oxyzn, oabcd);
#endif  // R == 3
#if R == 2
    xyzn2abcd_2(oxyzn, oabcd);
#endif  // R == 2
    iabcd[pm.axis[0]] = oabcd[0];
    iabcd[pm.axis[1]] = oabcd[1];
    iabcd[pm.axis[2]] = oabcd[2];
    iabcd[pm.axis[3]] = oabcd[3];
#if R == 4
    abcd2xyzn_4(pm.iC, iabcd, ixyzn);
#endif  // R == 4
#if R == 3
    abcd2xyzn_3(iabcd, ixyzn);
#endif  // R == 3
#if R == 2
    abcd2xyzn_2(iabcd, ixyzn);
#endif  // R == 2
    r[n] = inTexture.read(uint2(ixyzn[0], ixyzn[1]), ixyzn[2])[ixyzn[3]];
  }
  outTexture.write(r, gid.xy, gid.z);
}

#endif
