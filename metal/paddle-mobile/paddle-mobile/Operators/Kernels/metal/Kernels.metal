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

// 占位函数, 啥也没干
kernel void place_holder(texture2d<half, access::read> inTexture [[texture(0)]],
                         texture2d_array<half, access::write> outTexture [[texture(1)]],
                         uint3 gid [[thread_position_in_grid]]) {
}

struct OutputDim {
  ushort width;
  ushort height;
  ushort strideX;
  ushort strideY;
};

kernel void resize(texture2d<half, access::read> inTexture [[texture(0)]],
                   texture2d_array<half, access::write> outTexture [[texture(1)]],
                   constant OutputDim &params [[buffer(0)]],
                   uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= outTexture.get_width() ||
      gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size()) return;
  
  constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
  const uint2 pos = gid.xy * uint2(params.strideX, params.strideY);
  const half4 input = inTexture.read(pos);
  outTexture.write(half4(input.x, input.y, input.z, input.w), gid.xy, gid.z);
}



//kernel void texture2d_to_2d_array(texture2d<half, access::read> inTexture [[texture(0)]],
//                               texture2d_array<half, access::write> outTexture [[texture(1)]],
//                               uint3 gid [[thread_position_in_grid]]) {
//    if (gid.x >= inTexture.get_width() ||
//        gid.y >= inTexture.get_height()){
//        return;
//    }
//    const half4 input = inTexture.read(gid.xy);
//    outTexture.write(input, gid.xy, 0);
//}

kernel void texture2d_to_2d_array(texture2d<float, access::read> inTexture [[texture(0)]],
                                  texture2d_array<float, access::write> outTexture [[texture(1)]],
                                  uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= inTexture.get_width() ||
      gid.y >= inTexture.get_height()){
    return;
  }
  const float4 input = inTexture.read(gid.xy);
  outTexture.write(input, gid.xy, 0);
}


kernel void texture2d_to_2d_array_half(texture2d<half, access::read> inTexture [[texture(0)]],
                                       texture2d_array<half, access::write> outTexture [[texture(1)]],
                                       uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= inTexture.get_width() ||
      gid.y >= inTexture.get_height()){
    return;
  }
  const half4 input = inTexture.read(gid.xy);
  outTexture.write(input, gid.xy, 0);
}

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

struct TransposeParam {
  int iC;
  int oC;
  int axis[4];
};

kernel void transpose(texture2d_array<float, access::read> inTexture [[texture(0)]],
                      texture2d_array<float, access::write> outTexture [[texture(1)]],
                      constant TransposeParam &pm [[buffer(0)]],
                      uint3 gid [[thread_position_in_grid]]) {
  

  if ((pm.axis[0] == 0) && (pm.axis[1] == 1) && (pm.axis[2] == 2) && (pm.axis[3] == 3)) {
    // do nothing
    float4 r = inTexture.read(gid.xy, gid.z);
    outTexture.write(r, gid.xy, gid.z);
  } else {
    float4 r;
    for (int n = 0; n < 4; n++) {
      int ixyzn[] = {int(gid.x), int(gid.y), int(gid.z), n};
      int iabcd[4], oabcd[4], oxyzn[4];
      xyzn2abcd(pm.oC, ixyzn, iabcd);
      oabcd[pm.axis[0]] = iabcd[0];
      oabcd[pm.axis[1]] = iabcd[1];
      oabcd[pm.axis[2]] = iabcd[2];
      oabcd[pm.axis[3]] = iabcd[3];
      abcd2xyzn(pm.iC, oabcd, oxyzn);
      float4 rt = inTexture.read(uint2(oxyzn[0], oxyzn[1]), oxyzn[2]);
      r[n] = rt[oxyzn[3]];
    }
    outTexture.write(r, gid.xy, gid.z);
  }
}
