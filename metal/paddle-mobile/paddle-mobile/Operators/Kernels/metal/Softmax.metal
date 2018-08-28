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

#include <metal_stdlib>
using namespace metal;

struct SoftmaxParam {
  int N;
  int K;
};

kernel void softmax(texture2d_array<float, access::read> inTexture [[texture(0)]],
                    texture2d_array<float, access::write> outTexture [[texture(1)]],
                    constant SoftmaxParam &sp [[buffer(0)]],
                    uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= outTexture.get_width() ||
      gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size()) return;
//  int zsize = inTexture.get_array_size();
  float maxv = inTexture.read(gid.xy, 0)[0];
  int group = sp.K / 4;
  int remain = sp.K % 4;
  for (int z = 0; z < group; z++) {
    float4 r = inTexture.read(gid.xy, z);
    maxv = max(maxv, max(r[0], max(r[1], max(r[2], r[3]))));
  }
  if (remain > 0) {
    float4 r = inTexture.read(gid.xy, group);
    for (int i = 0; i < remain; i++) {
      maxv = max(maxv, r[i]);
    }
  }
  float4 rsum = {0, 0, 0, 0};
  for (int z = 0; z < group; z++) {
    float4 r = inTexture.read(gid.xy, z);
    rsum += exp(r - maxv);
  }
  float sum = rsum[0] + rsum[1] + rsum[2] + rsum[3];
  if (remain > 0) {
    float4 r = inTexture.read(gid.xy, group);
    for (int i = 0; i < remain; i++) {
      sum += exp(r[i] - maxv);
    }
  }
  float4 rr = inTexture.read(gid.xy, gid.z);
  rr = exp(rr - maxv) / sum;
  outTexture.write(rr, gid.xy, gid.z);
}
//
//kernel void softmax_half(texture2d_array<half, access::read> inTexture [[texture(0)]],
//                         texture2d_array<half, access::write> outTexture [[texture(1)]],
//                         uint3 gid [[thread_position_in_grid]]) {
//  if (gid.x >= outTexture.get_width() ||
//      gid.y >= outTexture.get_height() ||
//      gid.z >= outTexture.get_array_size()) return;
//  int zsize = inTexture.get_array_size();
//  half maxv = inTexture.read(uint2(0, 0), 0)[0];
//  for (int z = 0; z < zsize; z++) {
//    half4 r = inTexture.read(uint2(0, 0), z);
//    maxv = max(maxv, max(max(r[0], r[1]), max(r[2], r[3])));
//  }
//  float sum = 0;
//  for (int z = 0; z < zsize; z++) {
//    half4 r = inTexture.read(uint2(0, 0), z);
//    sum += exp(r[0] - maxv) + exp(r[1] - maxv) + exp(r[2] - maxv) + exp(r[3] - maxv);
//  }
//  half4 rr = inTexture.read(gid.xy, gid.z);
//  rr = exp(rr - maxv) / sum;
//  outTexture.write(rr, gid.xy, gid.z);
//}
