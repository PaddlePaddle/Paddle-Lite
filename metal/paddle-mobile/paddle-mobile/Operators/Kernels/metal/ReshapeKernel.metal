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

struct ReshapeParam {
  int32_t idim[4];
  int32_t itrans[4];
  int32_t odim[4];
  int32_t otrans[4];
};

//kernel void reshape(texture2d_array<float, access::read> inTexture [[texture(0)]],
//                    texture2d_array<float, access::write> outTexture [[texture(1)]],
//                    constant ReshapeParam &rp [[buffer(0)]],
//                    uint3 gid [[thread_position_in_grid]]) {
//  if (gid.x >= outTexture.get_width() ||
//      gid.y >= outTexture.get_height() ||
//      gid.z >= outTexture.get_array_size()) return;
//
//  int oxyzn[4] = {int(gid.x), int(gid.y), int(gid.z), 0}, oabcd[4], ixyzn[4];
//  ReshapeParam lrp = rp;
//  int oC = lrp.odim[lrp.otrans[3]];
//  int iC = lrp.idim[lrp.itrans[3]];
//  int count = lrp.odim[0] * lrp.odim[1] * lrp.odim[2] * lrp.odim[3];
//  float4 r;
//  for (int n = 0; n < 4; n++) {
//    oxyzn[3] = n;
//
//    //4  (gid.x gid.y, gid.z, 0~4)
//    xyzn2abcd(oC, oxyzn, oabcd);
//    int tabcd[4];
//    invtrans(lrp.otrans, oabcd, tabcd);
//    int index = abcd2index(lrp.odim, tabcd);
//    if (index < count) {
//      int c = index % 4;
//
//      int temp0 = index % (inTexture.get_array_size() * 4);
//      int slice = temp0 / 4;
//
//      int temp1 = index % (inTexture.get_array_size() * 4 * lrp.idim[2]);
//      int w = temp1 / (inTexture.get_array_size() * 4);
//
//      int h = index / (inTexture.get_array_size() * 4 * lrp.idim[2]);
//
////      index2abcd(lrp.idim, index, tabcd);
////      abcd2xyzn(iC, tabcd, ixyzn);
//      r[n] = inTexture.read(uint2(w, h), slice)[c];
//    } else {
//      r[n] = 0;
//    }
//  }
//  outTexture.write(r, gid.xy, gid.z);
//}




kernel void reshape(texture2d_array<float, access::read> inTexture [[texture(0)]],
                    texture2d_array<float, access::write> outTexture [[texture(1)]],
                    constant ReshapeParam &rp [[buffer(0)]],
                    uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= outTexture.get_width() ||
      gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size()) return;

  int oxyzn[4] = {int(gid.x), int(gid.y), int(gid.z), 0}, oabcd[4], ixyzn[4], iabcd[4];
  ReshapeParam lrp = rp;
  int oC = lrp.odim[lrp.otrans[3]];
  int iC = lrp.idim[lrp.itrans[3]];
  int count = lrp.odim[0] * lrp.odim[1] * lrp.odim[2] * lrp.odim[3];
  float4 r;
  for (int n = 0; n < 4; n++) {
    oxyzn[3] = n;
    xyzn2abcd(oC, oxyzn, oabcd);
    int tabcd[4];
    invtrans(lrp.otrans, oabcd, tabcd);
    int index = abcd2index(lrp.odim, tabcd);
    if (index < count) {
      index2abcd(lrp.idim, index, tabcd);
      trans(lrp.itrans, tabcd, iabcd);
      abcd2xyzn(iC, iabcd, ixyzn);
      r[n] = inTexture.read(uint2(ixyzn[0], ixyzn[1]), ixyzn[2])[ixyzn[3]];
    } else {
      r[n] = 0;
    }
  }
  outTexture.write(r, gid.xy, gid.z);
}

kernel void reshape_half(texture2d_array<half, access::read> inTexture [[texture(0)]],
                         texture2d_array<half, access::write> outTexture [[texture(1)]],
                         uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) return;

    half4 r = inTexture.read(uint2(0, 0), gid.x);
    outTexture.write(r, gid.xy, gid.z);
}

