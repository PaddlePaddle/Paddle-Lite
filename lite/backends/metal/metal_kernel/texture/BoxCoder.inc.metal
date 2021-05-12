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

#define FUNC(f, p) CONCAT2_(f, p)
#define VECTOR(p, n) CONCAT2(p, n)
kernel void FUNC(boxcoder,
                 P)(texture2d_array<P, access::read> priorBox [[texture(0)]],
                    texture2d_array<P, access::read> priorBoxVar [[texture(1)]],
                    texture2d_array<P, access::read> targetBox [[texture(2)]],
                    texture2d_array<P, access::write> output [[texture(3)]],
                    uint3 gid [[thread_position_in_grid]]) {
  VECTOR(P, 4) p = priorBox.read(uint2(0, gid.x), gid.z);
  VECTOR(P, 4) pv = priorBoxVar.read(uint2(0, gid.x), gid.z);
  VECTOR(P, 4) t;
  t[0] = targetBox.read(uint2(0, gid.x), gid.z)[0];
  t[1] = targetBox.read(uint2(1, gid.x), gid.z)[0];
  t[2] = targetBox.read(uint2(2, gid.x), gid.z)[0];
  t[3] = targetBox.read(uint2(3, gid.x), gid.z)[0];

  P px = (p.x + p.z) / 2;
  P py = (p.y + p.w) / 2;
  P pw = p.z - p.x;
  P ph = p.w - p.y;

  P tx = pv.x * t.x * pw + px;
  P ty = pv.y * t.y * ph + py;
  P tw = exp(pv.z * t.z) * pw;
  P th = exp(pv.w * t.w) * ph;

  VECTOR(P, 4) r;
  r.x = tx - tw / 2;
  r.y = ty - th / 2;
  r.z = tx + tw / 2;
  r.w = ty + th / 2;

  output.write(r, gid.xy, gid.z);
}

#endif
