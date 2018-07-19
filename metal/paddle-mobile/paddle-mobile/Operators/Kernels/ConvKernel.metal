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


//kernel void conv_add_batch_norm_relu_3x3(texture2d_array<half, access::sample> inTexture [[texture(0)]],
//                                        texture2d_array<half, access::write> outTexture [[texture(1)]],
//                                         constant MetalConvParam &param [[buffer(0)]],
//                                         const device half4 *weights [[buffer(1)]],
//                                         const device half4 *biase [[buffer(2)]],
//                                         const device half4 *new_scale [[buffer(3)]],
//                                         const device half4 *new_biase [[buffer(4)]],
//                                         uint3 gid [[thread_position_in_grid]]) {
//
//    if (gid.x >= outTexture.get_width() ||
//        gid.y >= outTexture.get_height() ||
//        gid.z >= outTexture.get_array_size()) {
//        return;
//    }
//
//    short2 posInInput = short2(gid.xy) + short2(param.offsetX, param.offsetY);
//    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
//    const uint wightSliceCount = 36;
//    uint weithTo = gid.z * wightSliceCount * inTexture.get_array_size();
//    half4 output = 0.0;
//    for (uint i = 0; i < inTexture.get_array_size(); ++i) {
//        half4 input[9];
//        input[0] = inTexture.sample(sample, float2(posInInput.x - 1, posInInput.y - 1), i);
//        input[1] = inTexture.sample(sample, float2(posInInput.x, posInInput.y - 1), i);
//        input[2] = inTexture.sample(sample, float2(posInInput.x + 1, posInInput.y - 1), i);
//        input[3] = inTexture.sample(sample, float2(posInInput.x - 1, posInInput.y), i);
//        input[4] = inTexture.sample(sample, float2(posInInput.x, posInInput.y), i);
//        input[5] = inTexture.sample(sample, float2(posInInput.x + 1, posInInput.y), i);
//        input[6] = inTexture.sample(sample, float2(posInInput.x - 1, posInInput.y + 1), i);
//        input[7] = inTexture.sample(sample, float2(posInInput.x, posInInput.y + 1), i);
//        input[8] = inTexture.sample(sample, float2(posInInput.x + 1, posInInput.y + 1), i);
//        for (int j = 0; j < 9; ++j) {
//            half4 weight = weights[weithTo + wightSliceCount * i + j * 4];
//            output += dot(input[j], weight);
//        }
//    }
//
//    output = fmax((output + biase[gid.z]) * new_scale[gid.z] + new_biase[gid.z], 0.0h);
//    outTexture.write(output, gid.xy, gid.z);
//
//}

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
    
    short2 posInInput = short2(gid.xy) + short2(param.offsetX, param.offsetY);
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
    
    short2 posInInput = short2(gid.xy) + short2(param.offsetX, param.offsetY);
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
    
    short2 posInInput = short2(gid.xy) + short2(param.offsetX, param.offsetY);
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
    short2 posInInput = short2(gid.xy) + short2(param.offsetX, param.offsetY);
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
    output = (output + biase[gid.z]) * new_scale[gid.z] + new_biase[gid.z];
    outTexture.write(output, gid.xy, gid.z);
}

