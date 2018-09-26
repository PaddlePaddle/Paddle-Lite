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

#define FUNC(f, p) CONCAT2_(f, p)
#define VECTOR(p, n) CONCAT2(p, n)

kernel void FUNC(softmax, P)(texture2d_array<P, access::read> inTexture [[texture(0)]],
                    texture2d_array<P, access::write> outTexture [[texture(1)]],
                    constant SoftmaxParam &sp [[buffer(0)]],
                    uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= outTexture.get_width() ||
      gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size()) return;
//  int zsize = inTexture.get_array_size();
  P maxv = inTexture.read(uint2(0, gid.y), 0)[0];
  int group = sp.K / 4;
  int remain = sp.K % 4;
  for (int x = 0; x < group; x++) {
    VECTOR(P, 4) r = inTexture.read(uint2(x, gid.y), 0);
    maxv = max(maxv, max(r[0], max(r[1], max(r[2], r[3]))));
  }
  if (remain > 0) {
    VECTOR(P, 4) r = inTexture.read(uint2(group, gid.y), 0);
    for (int i = 0; i < remain; i++) {
      maxv = max(maxv, r[i]);
    }
  }
  VECTOR(P, 4) rsum = {0, 0, 0, 0};
  for (int x = 0; x < group; x++) {
    VECTOR(P, 4) r = inTexture.read(uint2(x, gid.y), 0);
    rsum += exp(r - maxv);
  }
  P sum = rsum[0] + rsum[1] + rsum[2] + rsum[3];
  if (remain > 0) {
    VECTOR(P, 4) r = inTexture.read(uint2(group, gid.y), 0);
    for (int i = 0; i < remain; i++) {
      sum += exp(r[i] - maxv);
    }
  }
  VECTOR(P, 4) rr = inTexture.read(gid.xy, gid.z);
  rr = exp(rr - maxv) / sum;
  outTexture.write(rr, gid.xy, gid.z);
}

#endif
