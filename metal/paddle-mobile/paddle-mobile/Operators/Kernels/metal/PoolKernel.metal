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
#include "Common.metal"
using namespace metal;

struct PoolParam {
  int ksizeX;
  int ksizeY;
  int strideX;
  int strideY;
  int paddingX;
  int paddingY;
  int poolType;
};

kernel void pool(texture2d_array<float, access::read> inTexture [[texture(0)]],
                 texture2d_array<float, access::write> outTexture [[texture(1)]],
                 constant PoolParam &pm [[buffer(0)]],
                 uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= outTexture.get_width() ||
      gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size()) return;
  int xmin = gid.x * pm.strideX - pm.paddingX;
  int xmax = min(xmin + pm.ksizeX, int(inTexture.get_width()));
  xmin = max(xmin, 0);
  int ymin = gid.y * pm.strideX - pm.paddingX;
  int ymax = min(ymin + pm.ksizeX, int(inTexture.get_height()));
  ymin = max(ymin, 0);
  
  float4 r = 0;
  if (pm.poolType == 0) {
    r = inTexture.read(uint2(xmin, ymin), gid.z);
    for (int x = xmin; x < xmax; x++) {
      for (int y = ymin; y < ymax; y++) {
        r = fmax(r, inTexture.read(uint2(x, y), gid.z));
      }
    }
  } else if (pm.poolType == 1) {
    for (int x = xmin; x < xmax; x++) {
      for (int y = ymin; y < ymax; y++) {
        r += inTexture.read(uint2(x, y), gid.z);
      }
    }
    r /= pm.ksizeX * pm.ksizeY;
  }
  outTexture.write(r, gid.xy, gid.z);
}

kernel void pool_half(texture2d_array<half, access::read> inTexture [[texture(0)]],
                      texture2d_array<half, access::write> outTexture [[texture(1)]],
                      constant PoolParam &pm [[buffer(0)]],
                      uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= outTexture.get_width() ||
      gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size()) return;
  int xmin = gid.x * pm.strideX - pm.paddingX;
  int xmax = min(xmin + pm.ksizeX, int(inTexture.get_width()));
  xmin = max(xmin, 0);
  int ymin = gid.y * pm.strideX - pm.paddingX;
  int ymax = min(ymin + pm.ksizeX, int(inTexture.get_height()));
  ymin = max(ymin, 0);
  
  half4 r = 0;
  if (pm.poolType == 0) {
    r = inTexture.read(uint2(xmin, ymin), gid.z);
    for (int x = xmin; x < xmax; x++) {
      for (int y = ymin; y < ymax; y++) {
        r = fmax(r, inTexture.read(uint2(x, y), gid.z));
      }
    }
  } else if (pm.poolType == 1) {
    for (int x = xmin; x < xmax; x++) {
      for (int y = ymin; y < ymax; y++) {
        r += inTexture.read(uint2(x, y), gid.z);
      }
    }
    r /= pm.ksizeX * pm.ksizeY;
  }
  outTexture.write(r, gid.xy, gid.z);
}
