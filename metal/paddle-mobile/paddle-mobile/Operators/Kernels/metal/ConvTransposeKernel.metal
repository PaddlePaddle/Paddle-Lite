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

struct MetalConvTransposeParam{
  ushort kernelW;
  ushort kernelH;
  
  ushort strideX;
  ushort strideY;
  
  ushort paddingX;
  ushort paddingY;
  
  ushort dilationX;
  ushort dilationY;
};

kernel void conv_transpose(texture2d_array<float, access::sample> inTexture [[texture(0)]],
                           texture2d_array<float, access::write> outTexture [[texture(1)]],
                           constant MetalConvTransposeParam &param [[buffer(0)]],
                           const device float4 *weights [[buffer(1)]],
                           uint3 gid [[thread_position_in_grid]]){
  if (gid.x >= outTexture.get_width() ||
      gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size()) {
    return;
  }
  
  int input_array_size = inTexture.get_array_size();
  
  uint kernel_one_output_slice = input_array_size * param.kernelW * param.kernelH;
  
  uint kernel_stride_z = gid.z * 4 * (kernel_one_output_slice);
  
  constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
  
  float4 output;
  
  for (int w = 0; w < param.kernelW; ++w) {
    int input_x = (gid.x - w * param.dilationX + param.paddingX) / param.strideX;
    if (input_x < 0 || input_x >= int(inTexture.get_width())) {
      continue;
    }
    
    for (int h = 0; h < param.kernelH; ++h) {
      int input_y = (gid.y - h * param.dilationY + param.paddingY) / param.strideY;
      if (input_y < 0 || input_y >= int(inTexture.get_height())) {
        continue;
      }
      
      uint kernel_index = (w * param.kernelH + h) * inTexture.get_array_size();
      
      for (int slice = 0; slice < input_array_size; ++slice) {
        
        float4 input;
        float4 kernel_slice = weights[kernel_stride_z + 0 * kernel_one_output_slice + kernel_index + slice];
        float4 kernel_slice1 = weights[kernel_stride_z + 1 * kernel_one_output_slice + kernel_index + slice];
        
        float4 kernel_slice2 = weights[kernel_stride_z + 2 * kernel_one_output_slice + kernel_index + slice];
        
        float4 kernel_slice3 = weights[kernel_stride_z + 3 * kernel_one_output_slice + kernel_index + slice];
        
        input = inTexture.sample(sample, float2(input_x,    input_x), slice);
        output.x += dot(input, kernel_slice);
        output.x += dot(input, kernel_slice1);
        output.x += dot(input, kernel_slice2);
        output.x += dot(input, kernel_slice3);
      }
    }
  }
  
  outTexture.write(output, gid.xy, gid.z);
}

