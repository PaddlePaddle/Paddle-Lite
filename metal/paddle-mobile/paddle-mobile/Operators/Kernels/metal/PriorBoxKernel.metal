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

struct PriorBoxMetalParam {
  float offset;
  float stepWidth;
  float stepHeight;
  float minSize;
  float maxSize;
  float imageWidth;
  float imageHeight;
  
  bool clip;
  
  uint numPriors;
  uint aspecRatiosSize;
  uint minSizeSize;
  uint maxSizeSize;
};

kernel void prior_box(texture2d_array<float, access::read> inTexture [[texture(0)]],
                      texture2d_array<float, access::write> outBoxTexture [[texture(1)]],
                      texture2d_array<float, access::write> varianceTexture [[texture(2)]],
                      constant PriorBoxMetalParam &param [[buffer(0)]],
                      const device float *aspect_ratios [[buffer(1)]],
                      const device float4 *variances [[buffer(2)]],
                      uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= outBoxTexture.get_width() ||
      gid.y >= outBoxTexture.get_height() ||
      gid.z >= outBoxTexture.get_array_size()) return;
  
  float center_x = (gid.x + param.offset) * param.stepWidth;
  float center_y = (gid.y + param.offset) * param.stepHeight;
  
  float box_width, box_height;
  
  if (gid.z < param.aspecRatiosSize) {
    float ar = aspect_ratios[gid.z];
    box_width = param.minSize * sqrt(ar) / 2;
    box_height = param.minSize / sqrt(ar) / 2;
    float4 box;
    box.x = (center_x - box_width) / param.imageWidth;
    box.y = (center_y - box_height) / param.imageHeight;
    box.z = (center_x + box_width) / param.imageWidth;
    box.w = (center_y + box_height) / param.imageHeight;
    
    float4 res;
    if (param.clip) {
      res = fmin(fmax(box, 0.0), 1.0);
    } else {
      res = box;
    }
    
    outBoxTexture.write(res, gid.xy, gid.z);
  } else if (gid.z >= param.aspecRatiosSize) {
    if (param.maxSizeSize > 0) {
      box_width = box_height = sqrt(param.minSize * param.maxSize) / 2;
      float4 max_box;
      max_box.x = (center_x - box_width) / param.imageWidth;
      max_box.y = (center_y - box_height) / param.imageHeight;
      max_box.z = (center_x + box_width) / param.imageWidth;
      max_box.w = (center_y + box_height) / param.imageHeight;

      float4 res;
      if (param.clip) {
        res = min(max(max_box, 0.0), 1.0);
      } else {
        res = max_box;
      }
      outBoxTexture.write(max_box, gid.xy, gid.z);
    }
  }
  
  float4 variance = variances[0];
  if (gid.z < param.numPriors) {
    float4 variances_output;
    variances_output.x = variance.x;
    variances_output.y = variance.y;
    variances_output.z = variance.z;
    variances_output.w = variance.w;
    varianceTexture.write(variances_output, gid.xy, gid.z);
  }
}

