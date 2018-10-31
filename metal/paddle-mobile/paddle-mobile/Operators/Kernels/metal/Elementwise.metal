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

struct ElementwiseAddParam {
  int32_t fast;
  int32_t axis;
  int32_t ylen;
  int32_t xdim[4];
  int32_t xtrans[4];
  int32_t ydim[4];
  int32_t ytrans[4];
};

kernel void elementwise_add(texture2d_array<float, access::read> inputX [[texture(0)]],
                            texture2d_array<float, access::read> inputY [[texture(1)]],
                            texture2d_array<float, access::write> outTexture [[texture(2)]],
                            constant ElementwiseAddParam &pm [[buffer(0)]],
                            uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= outTexture.get_width() ||
      gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size()) return;
  float4 rx, ry;

  if (pm.fast == 1) {
    rx = inputX.read(gid.xy, gid.z);
    ry = inputY.read(gid.xy, gid.z);
  } else {
    rx = inputX.read(gid.xy, gid.z);
    int32_t x_xyzn[4] = {int32_t(gid.x), int32_t(gid.y), int32_t(gid.z), 0}, x_abcd[4], t_abcd[4];
    int32_t y_abcd[4] = {0, 0, 0, 0}, y_xyzn[4];
    int32_t xtrans[4] = {pm.xtrans[0], pm.xtrans[1], pm.xtrans[2], pm.xtrans[3]};
    int32_t ytrans[4] = {pm.ytrans[0], pm.ytrans[1], pm.ytrans[2], pm.ytrans[3]};
    int32_t yshift = 4 - pm.ylen - pm.axis;
    for (int n = 0; n < 4; n++) {
      x_xyzn[3] = n;
      xyzn2abcd(pm.xdim[3], x_xyzn, x_abcd);
      invtrans(xtrans, x_abcd, t_abcd);
      for (int k = pm.axis; k < (pm.axis + pm.ylen); k++) {
        y_abcd[yshift+k] = t_abcd[k];
      }
      trans(ytrans, y_abcd, t_abcd);
      abcd2xyzn(pm.ydim[3], t_abcd, y_xyzn);
      ry[n] = inputY.read(uint2(y_xyzn[0], y_xyzn[1]), y_xyzn[2])[y_xyzn[3]];
    }
  }
  float4 r = rx + ry;
  outTexture.write(r, gid.xy, gid.z);
}

kernel void elementwise_add_half(texture2d_array<half, access::read> inputX [[texture(0)]],
                            texture2d_array<half, access::read> inputY [[texture(1)]],
                            texture2d_array<half, access::write> outTexture [[texture(2)]],
                            constant ElementwiseAddParam &pm [[buffer(0)]],
                            uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= outTexture.get_width() ||
      gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size()) return;
  half4 rx, ry;

  if (pm.fast == 1) {
    rx = inputX.read(gid.xy, gid.z);
    ry = inputY.read(gid.xy, gid.z);
  } else {
    rx = inputX.read(gid.xy, gid.z);
    int32_t x_xyzn[4] = {int32_t(gid.x), int32_t(gid.y), int32_t(gid.z), 0}, x_abcd[4], t_abcd[4];
    int32_t y_abcd[4] = {0, 0, 0, 0}, y_xyzn[4];
    int32_t xtrans[4] = {pm.xtrans[0], pm.xtrans[1], pm.xtrans[2], pm.xtrans[3]};
    int32_t ytrans[4] = {pm.ytrans[0], pm.ytrans[1], pm.ytrans[2], pm.ytrans[3]};
    int32_t yshift = 4 - pm.ylen - pm.axis;
    for (int n = 0; n < 4; n++) {
      x_xyzn[3] = n;
      xyzn2abcd(pm.xdim[3], x_xyzn, x_abcd);
      invtrans(xtrans, x_abcd, t_abcd);
      for (int k = pm.axis; k < (pm.axis + pm.ylen); k++) {
        y_abcd[yshift+k] = t_abcd[k];
      }
      trans(ytrans, y_abcd, t_abcd);
      abcd2xyzn(pm.ydim[3], t_abcd, y_xyzn);
      ry[n] = inputY.read(uint2(y_xyzn[0], y_xyzn[1]), y_xyzn[2])[y_xyzn[3]];
    }
  }
  half4 r = rx + ry;
  outTexture.write(r, gid.xy, gid.z);
}
