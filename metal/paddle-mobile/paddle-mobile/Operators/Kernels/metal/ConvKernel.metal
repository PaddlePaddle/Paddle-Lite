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

struct MetalConvParam {
    short offsetX;
    short offsetY;
    short offsetZ;
    ushort strideX;
    ushort strideY;
};


kernel void conv_add_batch_norm_relu_1x1_half(texture2d_array<half, access::sample> inTexture [[texture(0)]],
                                         texture2d_array<half, access::write> outTexture [[texture(1)]],
                                         constant MetalConvParam &param [[buffer(0)]],
                                         const device half4 *weights [[buffer(1)]],
                                         const device half4 *biase [[buffer(2)]],
                                         const device float4 *new_scale [[buffer(3)]],
                                         const device float4 *new_biase [[buffer(4)]],
                                         uint3 gid [[thread_position_in_grid]]) {
    
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }
    
    
    ushort2 stride = ushort2(param.strideX, param.strideY);
    ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);
    
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    const uint kernelHXW = 1;
    
    uint input_arr_size = inTexture.get_array_size();
    uint weithTo = gid.z * kernelHXW * input_arr_size * 4;
    
    half4 output = half4(0.0);
    
    half4 input;
    for (uint i = 0; i < input_arr_size; ++i) {
        input = inTexture.sample(sample, float2(posInInput.x, posInInput.y), i);
        half4 weight_x = weights[weithTo + 0 * kernelHXW * input_arr_size  + i];
        output.x += dot(input, weight_x);
        
        half4 weight_y = weights[weithTo + 1 * kernelHXW * input_arr_size  + i];
        output.y += dot(input, weight_y);
        
        half4 weight_z = weights[weithTo + 2 * kernelHXW * input_arr_size  + i];
        output.z += dot(input, weight_z);
        
        half4 weight_w = weights[weithTo + 3 * kernelHXW * input_arr_size + i];
        output.w += dot(input, weight_w);
    }
    
    output = half4(fmax((float4(output) + float4(biase[gid.z])) * new_scale[gid.z] + new_biase[gid.z], 0.0));
    outTexture.write(output, gid.xy, gid.z);
}

kernel void conv_add_batch_norm_relu_3x3_half(texture2d_array<half, access::sample> inTexture [[texture(0)]],
                                         texture2d_array<half, access::write> outTexture [[texture(1)]],
                                         constant MetalConvParam &param [[buffer(0)]],
                                         const device half4 *weights [[buffer(1)]],
                                         const device half4 *biase [[buffer(2)]],
                                         const device float4 *new_scale [[buffer(3)]],
                                         const device float4 *new_biase [[buffer(4)]],
                                         uint3 gid [[thread_position_in_grid]]) {
    
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }
    
    ushort2 stride = ushort2(param.strideX, param.strideY);
    const ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);
    
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    const uint kernelHXW = 9;
    uint input_arr_size = inTexture.get_array_size();
    uint weithTo = gid.z * kernelHXW * input_arr_size * 4;
    
    half4 output = half4(0.0);
    
    half4 input[9];
    for (uint i = 0; i < input_arr_size; ++i) {
        input[0] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y - 1), i);
        input[1] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y - 1), i);
        input[2] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y - 1), i);
        input[3] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y), i);
        input[4] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y), i);
        input[5] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y), i);
        input[6] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y + 1), i);
        input[7] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y + 1), i);
        input[8] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y + 1), i);
        for (int j = 0; j < 9; ++j) {
            half4 weight_x = weights[weithTo + 0 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.x += dot(input[j], weight_x);
            
            half4 weight_y = weights[weithTo + 1 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.y += dot(input[j], weight_y);
            
            half4 weight_z = weights[weithTo + 2 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.z += dot(input[j], weight_z);
            
            half4 weight_w = weights[weithTo + 3 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.w += dot(input[j], weight_w);
        }
    }
    output = half4(fmax((float4(output) + float4(biase[gid.z])) * new_scale[gid.z] + new_biase[gid.z], 0.0));
    outTexture.write(output, gid.xy, gid.z);
}

kernel void conv_add_1x1_half(texture2d_array<half, access::sample> inTexture [[texture(0)]],
                         texture2d_array<half, access::write> outTexture [[texture(1)]],
                         constant MetalConvParam &param [[buffer(0)]],
                         const device half4 *weights [[buffer(1)]],
                         const device half4 *biase [[buffer(2)]],
                         uint3 gid [[thread_position_in_grid]]) {
    
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }
    
    ushort2 stride = ushort2(param.strideX, param.strideY);
    ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);
    
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    const uint kernelHXW = 1;
    
    uint input_arr_size = inTexture.get_array_size();
    uint weithTo = gid.z * kernelHXW * input_arr_size * 4;
    
    half4 output = half4(0.0);
    
    half4 input;
    for (uint i = 0; i < input_arr_size; ++i) {
        input = inTexture.sample(sample, float2(posInInput.x, posInInput.y), i);
        half4 weight_x = weights[weithTo + 0 * kernelHXW * input_arr_size  + i];
        output.x += dot(input, weight_x);
        
        half4 weight_y = weights[weithTo + 1 * kernelHXW * input_arr_size  + i];
        output.y += dot(input, weight_y);
        
        half4 weight_z = weights[weithTo + 2 * kernelHXW * input_arr_size  + i];
        output.z += dot(input, weight_z);
        
        half4 weight_w = weights[weithTo + 3 * kernelHXW * input_arr_size + i];
        output.w += dot(input, weight_w);
    }
    output = output + biase[gid.z];
    outTexture.write(output, gid.xy, gid.z);
}

kernel void depthwise_conv_add_batch_norm_relu_3x3_half(texture2d_array<half, access::sample> inTexture [[texture(0)]],
                                                   texture2d_array<half, access::write> outTexture [[texture(1)]],
                                                   constant MetalConvParam &param [[buffer(0)]],
                                                   const device half *weights [[buffer(1)]],
                                                   const device half4 *biase [[buffer(2)]],
                                                   const device float4 *new_scale [[buffer(3)]],
                                                   const device float4 *new_biase [[buffer(4)]],
                                                   uint3 gid [[thread_position_in_grid]]) {
    
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }
    uint output_slice = gid.z;
    ushort2 stride = ushort2(param.strideX, param.strideY);
    ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    const uint kernelHXW = 9;
    uint weithTo = gid.z * kernelHXW * 4;
    half4 output = half4(0.0);
    half4 inputs[9];
    inputs[0] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y - 1), output_slice);
    inputs[1] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y - 1), output_slice);
    inputs[2] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y - 1), output_slice);
    inputs[3] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y), output_slice);
    inputs[4] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y), output_slice);
    inputs[5] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y), output_slice);
    inputs[6] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y + 1), output_slice);
    inputs[7] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y + 1), output_slice);
    inputs[8] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y + 1), output_slice);
    for (int j = 0; j < 9; ++j) {
        half4 input = inputs[j];
        output.x += input.x * weights[weithTo + 0 * kernelHXW + j];
        output.y += input.y * weights[weithTo + 1 * kernelHXW + j];
        output.z += input.z * weights[weithTo + 2 * kernelHXW + j];
        output.w += input.w * weights[weithTo + 3 * kernelHXW + j];
    }
    output = half4(fmax((float4(output) + float4(biase[gid.z])) * new_scale[gid.z] + new_biase[gid.z], 0.0));
    outTexture.write(output, gid.xy, gid.z);
}


/*---------------------------------------------*/



kernel void conv_add_batch_norm_relu_1x1(texture2d_array<float, access::sample> inTexture [[texture(0)]],
                                         texture2d_array<float, access::write> outTexture [[texture(1)]],
                                         constant MetalConvParam &param [[buffer(0)]],
                                         const device float4 *weights [[buffer(1)]],
                                         const device float4 *biase [[buffer(2)]],
                                         const device float4 *new_scale [[buffer(3)]],
                                         const device float4 *new_biase [[buffer(4)]],
                                         uint3 gid [[thread_position_in_grid]]) {
    
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }
    
    ushort2 stride = ushort2(param.strideX, param.strideY);
    ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);
    
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    const uint kernelHXW = 1;
    
    uint input_arr_size = inTexture.get_array_size();
    uint weithTo = gid.z * kernelHXW * input_arr_size * 4;
    
    float4 output = float4(0.0);
    
    float4 input;
    for (uint i = 0; i < input_arr_size; ++i) {
        input = inTexture.sample(sample, float2(posInInput.x, posInInput.y), i);
        float4 weight_x = weights[weithTo + 0 * kernelHXW * input_arr_size  + i];
        output.x += dot(input, weight_x);
        
        float4 weight_y = weights[weithTo + 1 * kernelHXW * input_arr_size  + i];
        output.y += dot(input, weight_y);
        
        float4 weight_z = weights[weithTo + 2 * kernelHXW * input_arr_size  + i];
        output.z += dot(input, weight_z);
        
        float4 weight_w = weights[weithTo + 3 * kernelHXW * input_arr_size + i];
        output.w += dot(input, weight_w);
    }
    output = fmax((output + biase[gid.z]) * new_scale[gid.z] + new_biase[gid.z], 0.0);
    outTexture.write(output, gid.xy, gid.z);
}

kernel void conv_add_batch_norm_relu_3x3(texture2d_array<float, access::sample> inTexture [[texture(0)]],
                                         texture2d_array<float, access::write> outTexture [[texture(1)]],
                                         constant MetalConvParam &param [[buffer(0)]],
                                         const device float4 *weights [[buffer(1)]],
                                         const device float4 *biase [[buffer(2)]],
                                         const device float4 *new_scale [[buffer(3)]],
                                         const device float4 *new_biase [[buffer(4)]],
                                         uint3 gid [[thread_position_in_grid]]) {
    
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }
    
    ushort2 stride = ushort2(param.strideX, param.strideY);
    const ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);
    
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    const uint kernelHXW = 9;
    uint input_arr_size = inTexture.get_array_size();
    uint weithTo = gid.z * kernelHXW * input_arr_size * 4;
    
    float4 output = float4(0.0);
    
    float4 input[9];
    for (uint i = 0; i < input_arr_size; ++i) {
        input[0] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y - 1), i);
        input[1] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y - 1), i);
        input[2] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y - 1), i);
        input[3] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y), i);
        input[4] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y), i);
        input[5] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y), i);
        input[6] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y + 1), i);
        input[7] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y + 1), i);
        input[8] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y + 1), i);
        for (int j = 0; j < 9; ++j) {
            float4 weight_x = weights[weithTo + 0 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.x += dot(input[j], weight_x);
            
            float4 weight_y = weights[weithTo + 1 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.y += dot(input[j], weight_y);
            
            float4 weight_z = weights[weithTo + 2 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.z += dot(input[j], weight_z);
            
            float4 weight_w = weights[weithTo + 3 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.w += dot(input[j], weight_w);
        }
    }
    output = fmax((output + biase[gid.z]) * new_scale[gid.z] + new_biase[gid.z], 0.0);
    outTexture.write(output, gid.xy, gid.z);
}

kernel void depthwise_conv_add_batch_norm_relu_3x3(texture2d_array<float, access::sample> inTexture [[texture(0)]],
                                         texture2d_array<float, access::write> outTexture [[texture(1)]],
                                         constant MetalConvParam &param [[buffer(0)]],
                                         const device float *weights [[buffer(1)]],
                                         const device float4 *biase [[buffer(2)]],
                                         const device float4 *new_scale [[buffer(3)]],
                                         const device float4 *new_biase [[buffer(4)]],
                                         uint3 gid [[thread_position_in_grid]]) {
    
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }
    uint output_slice = gid.z;
    ushort2 stride = ushort2(param.strideX, param.strideY);
    ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    const uint kernelHXW = 9;
    uint weithTo = gid.z * kernelHXW * 4;
    float4 output = float4(0.0);
    float4 inputs[9];
    inputs[0] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y - 1), output_slice);
    inputs[1] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y - 1), output_slice);
    inputs[2] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y - 1), output_slice);
    inputs[3] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y), output_slice);
    inputs[4] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y), output_slice);
    inputs[5] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y), output_slice);
    inputs[6] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y + 1), output_slice);
    inputs[7] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y + 1), output_slice);
    inputs[8] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y + 1), output_slice);
    for (int j = 0; j < 9; ++j) {
        float4 input = inputs[j];
        output.x += input.x * weights[weithTo + 0 * kernelHXW + j];
        output.y += input.y * weights[weithTo + 1 * kernelHXW + j];
        output.z += input.z * weights[weithTo + 2 * kernelHXW + j];
        output.w += input.w * weights[weithTo + 3 * kernelHXW + j];
    }
    output = fmax((output + biase[gid.z]) * new_scale[gid.z] + new_biase[gid.z], 0.0);
    outTexture.write(output, gid.xy, gid.z);
}

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


// conv
#pragma mark -- conv
kernel void conv_3x3(texture2d_array<float, access::sample> inTexture [[texture(0)]],
                     texture2d_array<float, access::write> outTexture [[texture(1)]],
                     constant MetalConvParam &param [[buffer(0)]],
                     const device float4 *weights [[buffer(1)]],
                     uint3 gid [[thread_position_in_grid]]) {
  
  if (gid.x >= outTexture.get_width() ||
      gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size()) {
    return;
  }
  
  ushort2 stride = ushort2(param.strideX, param.strideY);
  const ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);
  
  constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
  const uint kernelHXW = 9;
  uint input_arr_size = inTexture.get_array_size();
  uint weithTo = gid.z * kernelHXW * input_arr_size * 4;
  
  float4 output = float4(0.0);
  
  float4 input[9];
  for (uint i = 0; i < input_arr_size; ++i) {
    input[0] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y - 1), i);
    input[1] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y - 1), i);
    input[2] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y - 1), i);
    input[3] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y), i);
    input[4] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y), i);
    input[5] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y), i);
    input[6] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y + 1), i);
    input[7] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y + 1), i);
    input[8] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y + 1), i);
    for (int j = 0; j < 9; ++j) {
      float4 weight_x = weights[weithTo + 0 * kernelHXW * input_arr_size + j * input_arr_size + i];
      output.x += dot(input[j], weight_x);
      
      float4 weight_y = weights[weithTo + 1 * kernelHXW * input_arr_size + j * input_arr_size + i];
      output.y += dot(input[j], weight_y);
      
      float4 weight_z = weights[weithTo + 2 * kernelHXW * input_arr_size + j * input_arr_size + i];
      output.z += dot(input[j], weight_z);
      
      float4 weight_w = weights[weithTo + 3 * kernelHXW * input_arr_size + j * input_arr_size + i];
      output.w += dot(input[j], weight_w);
    }
  }
  outTexture.write(output, gid.xy, gid.z);
}

kernel void depthwise_conv_3x3(texture2d_array<float, access::sample> inTexture [[texture(0)]],
                               texture2d_array<float, access::write> outTexture [[texture(1)]],
                               constant MetalConvParam &param [[buffer(0)]],
                               const device float *weights [[buffer(1)]],
                               uint3 gid [[thread_position_in_grid]]) {
  
  if (gid.x >= outTexture.get_width() ||
      gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size()) {
    return;
  }
  uint output_slice = gid.z;
  ushort2 stride = ushort2(param.strideX, param.strideY);
  ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);
  constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
  const uint kernelHXW = 9;
  uint weithTo = gid.z * kernelHXW * 4;
  float4 output = float4(0.0);
  float4 inputs[9];
  inputs[0] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y - 1), output_slice);
  inputs[1] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y - 1), output_slice);
  inputs[2] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y - 1), output_slice);
  inputs[3] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y), output_slice);
  inputs[4] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y), output_slice);
  inputs[5] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y), output_slice);
  inputs[6] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y + 1), output_slice);
  inputs[7] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y + 1), output_slice);
  inputs[8] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y + 1), output_slice);
  for (int j = 0; j < 9; ++j) {
    float4 input = inputs[j];
    output.x += input.x * weights[weithTo + 0 * kernelHXW + j];
    output.y += input.y * weights[weithTo + 1 * kernelHXW + j];
    output.z += input.z * weights[weithTo + 2 * kernelHXW + j];
    output.w += input.w * weights[weithTo + 3 * kernelHXW + j];
  }
  outTexture.write(output, gid.xy, gid.z);
}

kernel void conv_1x1(texture2d_array<float, access::sample> inTexture [[texture(0)]],
                         texture2d_array<float, access::write> outTexture [[texture(1)]],
                         constant MetalConvParam &param [[buffer(0)]],
                         const device float4 *weights [[buffer(1)]],
                         uint3 gid [[thread_position_in_grid]]) {
  
  if (gid.x >= outTexture.get_width() ||
      gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size()) {
    return;
  }
  
  ushort2 stride = ushort2(param.strideX, param.strideY);
  ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);
  
  constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
  const uint kernelHXW = 1;
  
  uint input_arr_size = inTexture.get_array_size();
  uint weithTo = gid.z * kernelHXW * input_arr_size * 4;
  
  float4 output = float4(0.0);
  
  float4 input;
  for (uint i = 0; i < input_arr_size; ++i) {
    input = inTexture.sample(sample, float2(posInInput.x, posInInput.y), i);
    float4 weight_x = weights[weithTo + 0 * kernelHXW * input_arr_size  + i];
    output.x += dot(input, weight_x);
    
    float4 weight_y = weights[weithTo + 1 * kernelHXW * input_arr_size  + i];
    output.y += dot(input, weight_y);
    
    float4 weight_z = weights[weithTo + 2 * kernelHXW * input_arr_size  + i];
    output.z += dot(input, weight_z);
    
    float4 weight_w = weights[weithTo + 3 * kernelHXW * input_arr_size + i];
    output.w += dot(input, weight_w);
  }
  outTexture.write(output, gid.xy, gid.z);
}

#pragma mark - convAdd
kernel void conv_add_1x1(texture2d_array<float, access::sample> inTexture [[texture(0)]],
                         texture2d_array<float, access::write> outTexture [[texture(1)]],
                         constant MetalConvParam &param [[buffer(0)]],
                         const device float4 *weights [[buffer(1)]],
                         const device float4 *biase [[buffer(2)]],
                         uint3 gid [[thread_position_in_grid]]) {
  
  if (gid.x >= outTexture.get_width() ||
      gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size()) {
    return;
  }
  
  ushort2 stride = ushort2(param.strideX, param.strideY);
  ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);
  
  constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
  const uint kernelHXW = 1;
  
  uint input_arr_size = inTexture.get_array_size();
  uint weithTo = gid.z * kernelHXW * input_arr_size * 4;
  
  float4 output = float4(0.0);
  
  float4 input;
  for (uint i = 0; i < input_arr_size; ++i) {
    input = inTexture.sample(sample, float2(posInInput.x, posInInput.y), i);
    float4 weight_x = weights[weithTo + 0 * kernelHXW * input_arr_size  + i];
    output.x += dot(input, weight_x);
    
    float4 weight_y = weights[weithTo + 1 * kernelHXW * input_arr_size  + i];
    output.y += dot(input, weight_y);
    
    float4 weight_z = weights[weithTo + 2 * kernelHXW * input_arr_size  + i];
    output.z += dot(input, weight_z);
    
    float4 weight_w = weights[weithTo + 3 * kernelHXW * input_arr_size + i];
    output.w += dot(input, weight_w);
  }
  output = output + biase[gid.z];
  outTexture.write(output, gid.xy, gid.z);
}

kernel void conv_add_3x3(texture2d_array<float, access::sample> inTexture [[texture(0)]],
                                         texture2d_array<float, access::write> outTexture [[texture(1)]],
                                         constant MetalConvParam &param [[buffer(0)]],
                                         const device float4 *weights [[buffer(1)]],
                                         const device float4 *biase [[buffer(2)]],
                                         const device float4 *new_scale [[buffer(3)]],
                                         const device float4 *new_biase [[buffer(4)]],
                                         uint3 gid [[thread_position_in_grid]]) {
  
  if (gid.x >= outTexture.get_width() ||
      gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size()) {
    return;
  }
  
  ushort2 stride = ushort2(param.strideX, param.strideY);
  const ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);
  
  constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
  const uint kernelHXW = 9;
  uint input_arr_size = inTexture.get_array_size();
  uint weithTo = gid.z * kernelHXW * input_arr_size * 4;
  
  float4 output = float4(0.0);
  
  float4 input[9];
  for (uint i = 0; i < input_arr_size; ++i) {
    input[0] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y - 1), i);
    input[1] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y - 1), i);
    input[2] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y - 1), i);
    input[3] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y), i);
    input[4] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y), i);
    input[5] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y), i);
    input[6] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y + 1), i);
    input[7] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y + 1), i);
    input[8] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y + 1), i);
    for (int j = 0; j < 9; ++j) {
      float4 weight_x = weights[weithTo + 0 * kernelHXW * input_arr_size + j * input_arr_size + i];
      output.x += dot(input[j], weight_x);
      
      float4 weight_y = weights[weithTo + 1 * kernelHXW * input_arr_size + j * input_arr_size + i];
      output.y += dot(input[j], weight_y);
      
      float4 weight_z = weights[weithTo + 2 * kernelHXW * input_arr_size + j * input_arr_size + i];
      output.z += dot(input[j], weight_z);
      
      float4 weight_w = weights[weithTo + 3 * kernelHXW * input_arr_size + j * input_arr_size + i];
      output.w += dot(input[j], weight_w);
    }
  }
  output = output + biase[gid.z];
  outTexture.write(output, gid.xy, gid.z);
}

kernel void depthwise_conv_add_3x3(texture2d_array<float, access::sample> inTexture [[texture(0)]],
                                                   texture2d_array<float, access::write> outTexture [[texture(1)]],
                                                   constant MetalConvParam &param [[buffer(0)]],
                                                   const device float *weights [[buffer(1)]],
                                                   const device float4 *biase [[buffer(2)]],
                                                   const device float4 *new_scale [[buffer(3)]],
                                                   const device float4 *new_biase [[buffer(4)]],
                                                   uint3 gid [[thread_position_in_grid]]) {
  
  if (gid.x >= outTexture.get_width() ||
      gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size()) {
    return;
  }
  uint output_slice = gid.z;
  ushort2 stride = ushort2(param.strideX, param.strideY);
  ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);
  constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
  const uint kernelHXW = 9;
  uint weithTo = gid.z * kernelHXW * 4;
  float4 output = float4(0.0);
  float4 inputs[9];
  inputs[0] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y - 1), output_slice);
  inputs[1] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y - 1), output_slice);
  inputs[2] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y - 1), output_slice);
  inputs[3] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y), output_slice);
  inputs[4] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y), output_slice);
  inputs[5] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y), output_slice);
  inputs[6] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y + 1), output_slice);
  inputs[7] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y + 1), output_slice);
  inputs[8] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y + 1), output_slice);
  for (int j = 0; j < 9; ++j) {
    float4 input = inputs[j];
    output.x += input.x * weights[weithTo + 0 * kernelHXW + j];
    output.y += input.y * weights[weithTo + 1 * kernelHXW + j];
    output.z += input.z * weights[weithTo + 2 * kernelHXW + j];
    output.w += input.w * weights[weithTo + 3 * kernelHXW + j];
  }
  output = output + biase[gid.z];
  outTexture.write(output, gid.xy, gid.z);
}

#pragma mark - conv bn relu
kernel void conv_batch_norm_relu_1x1(texture2d_array<float, access::sample> inTexture [[texture(0)]],
                                         texture2d_array<float, access::write> outTexture [[texture(1)]],
                                         constant MetalConvParam &param [[buffer(0)]],
                                         const device float4 *weights [[buffer(1)]],
                                         const device float4 *new_scale [[buffer(2)]],
                                         const device float4 *new_biase [[buffer(3)]],
                                         uint3 gid [[thread_position_in_grid]]) {
  
  if (gid.x >= outTexture.get_width() ||
      gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size()) {
    return;
  }
  
  ushort2 stride = ushort2(param.strideX, param.strideY);
  ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);
  
  constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
  const uint kernelHXW = 1;
  
  uint input_arr_size = inTexture.get_array_size();
  uint weithTo = gid.z * kernelHXW * input_arr_size * 4;
  
  float4 output = float4(0.0);
  
  float4 input;
  for (uint i = 0; i < input_arr_size; ++i) {
    input = inTexture.sample(sample, float2(posInInput.x, posInInput.y), i);
    float4 weight_x = weights[weithTo + 0 * kernelHXW * input_arr_size  + i];
    output.x += dot(input, weight_x);
    
    float4 weight_y = weights[weithTo + 1 * kernelHXW * input_arr_size  + i];
    output.y += dot(input, weight_y);
    
    float4 weight_z = weights[weithTo + 2 * kernelHXW * input_arr_size  + i];
    output.z += dot(input, weight_z);
    
    float4 weight_w = weights[weithTo + 3 * kernelHXW * input_arr_size + i];
    output.w += dot(input, weight_w);
  }
  output = fmax(output * new_scale[gid.z] + new_biase[gid.z], 0.0);
  outTexture.write(output, gid.xy, gid.z);
}

kernel void conv_batch_norm_relu_3x3(texture2d_array<float, access::sample> inTexture [[texture(0)]],
                                         texture2d_array<float, access::write> outTexture [[texture(1)]],
                                         constant MetalConvParam &param [[buffer(0)]],
                                         const device float4 *weights [[buffer(1)]],
                                         const device float4 *new_scale [[buffer(2)]],
                                         const device float4 *new_biase [[buffer(3)]],
                                         uint3 gid [[thread_position_in_grid]]) {
  
  if (gid.x >= outTexture.get_width() ||
      gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size()) {
    return;
  }
  
  ushort2 stride = ushort2(param.strideX, param.strideY);
  const ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);
  
  constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
  const uint kernelHXW = 9;
  uint input_arr_size = inTexture.get_array_size();
  uint weithTo = gid.z * kernelHXW * input_arr_size * 4;
  
  float4 output = float4(0.0);
  
  float4 input[9];
  for (uint i = 0; i < input_arr_size; ++i) {
    input[0] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y - 1), i);
    input[1] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y - 1), i);
    input[2] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y - 1), i);
    input[3] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y), i);
    input[4] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y), i);
    input[5] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y), i);
    input[6] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y + 1), i);
    input[7] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y + 1), i);
    input[8] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y + 1), i);
    for (int j = 0; j < 9; ++j) {
      float4 weight_x = weights[weithTo + 0 * kernelHXW * input_arr_size + j * input_arr_size + i];
      output.x += dot(input[j], weight_x);
      
      float4 weight_y = weights[weithTo + 1 * kernelHXW * input_arr_size + j * input_arr_size + i];
      output.y += dot(input[j], weight_y);
      
      float4 weight_z = weights[weithTo + 2 * kernelHXW * input_arr_size + j * input_arr_size + i];
      output.z += dot(input[j], weight_z);
      
      float4 weight_w = weights[weithTo + 3 * kernelHXW * input_arr_size + j * input_arr_size + i];
      output.w += dot(input[j], weight_w);
    }
  }
  output = fmax(output * new_scale[gid.z] + new_biase[gid.z], 0.0);
  outTexture.write(output, gid.xy, gid.z);
}

kernel void depthwise_conv_batch_norm_relu_3x3(texture2d_array<float, access::sample> inTexture [[texture(0)]],
                                                   texture2d_array<float, access::write> outTexture [[texture(1)]],
                                                   constant MetalConvParam &param [[buffer(0)]],
                                                   const device float *weights [[buffer(1)]],
                                                   const device float4 *new_scale [[buffer(2)]],
                                                   const device float4 *new_biase [[buffer(3)]],
                                                   uint3 gid [[thread_position_in_grid]]) {
  
  if (gid.x >= outTexture.get_width() ||
      gid.y >= outTexture.get_height() ||
      gid.z >= outTexture.get_array_size()) {
    return;
  }
  uint output_slice = gid.z;
  ushort2 stride = ushort2(param.strideX, param.strideY);
  ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);
  constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
  const uint kernelHXW = 9;
  uint weithTo = gid.z * kernelHXW * 4;
  float4 output = float4(0.0);
  float4 inputs[9];
  inputs[0] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y - 1), output_slice);
  inputs[1] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y - 1), output_slice);
  inputs[2] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y - 1), output_slice);
  inputs[3] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y), output_slice);
  inputs[4] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y), output_slice);
  inputs[5] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y), output_slice);
  inputs[6] = inTexture.sample(sample, float2(posInInput.x - 1,    posInInput.y + 1), output_slice);
  inputs[7] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y + 1), output_slice);
  inputs[8] = inTexture.sample(sample, float2(posInInput.x + 1,    posInInput.y + 1), output_slice);
  for (int j = 0; j < 9; ++j) {
    float4 input = inputs[j];
    output.x += input.x * weights[weithTo + 0 * kernelHXW + j];
    output.y += input.y * weights[weithTo + 1 * kernelHXW + j];
    output.z += input.z * weights[weithTo + 2 * kernelHXW + j];
    output.w += input.w * weights[weithTo + 3 * kernelHXW + j];
  }
  output = fmax(output * new_scale[gid.z] + new_biase[gid.z], 0.0);
  outTexture.write(output, gid.xy, gid.z);
}

