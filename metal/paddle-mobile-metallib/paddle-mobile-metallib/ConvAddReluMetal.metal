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

#pragma mark - convAdd
kernel void conv_add_relu_1x1(texture2d_array<float, access::sample> inTexture [[texture(0)]],
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
    
    float4 output = biase[gid.z];
    
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
    float4 relu = fmax(output, 0.0);
    outTexture.write(relu, gid.xy, gid.z);
}

kernel void conv_add_relu_3x3(texture2d_array<float, access::sample> inTexture [[texture(0)]],
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
    const ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);
    
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    
    const uint kernelHXW = 9;
    
    uint input_arr_size = inTexture.get_array_size();
    
    uint weithTo = gid.z * kernelHXW * input_arr_size * 4;
    
    float4 output = biase[gid.z];
    
    ushort dilation_x = param.dilationX;
    ushort dilation_y = param.dilationY;
    
    float4 input[9];
    
    for (uint i = 0; i < input_arr_size; ++i) {
        input[0] = inTexture.sample(sample, float2(posInInput.x - dilation_x, posInInput.y - dilation_y), i);
        
        input[1] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y - dilation_y), i);
        
        input[2] = inTexture.sample(sample, float2(posInInput.x + dilation_x, posInInput.y - dilation_y), i);
        
        input[3] = inTexture.sample(sample, float2(posInInput.x - dilation_x,    posInInput.y), i);
        
        input[4] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y), i);
        
        input[5] = inTexture.sample(sample, float2(posInInput.x + dilation_x,    posInInput.y), i);
        
        input[6] = inTexture.sample(sample, float2(posInInput.x - dilation_x,    posInInput.y + dilation_y), i);
        
        input[7] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y + dilation_y), i);
        
        input[8] = inTexture.sample(sample, float2(posInInput.x + dilation_x,    posInInput.y + dilation_y), i);
        
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
    float4 relu = fmax(output, 0.0);
    outTexture.write(relu, gid.xy, gid.z);
}

kernel void conv_add_relu_5x1(texture2d_array<float, access::sample> inTexture [[texture(0)]],
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
    const ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);
    
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    
    const uint kernelHXW = 5;
    
    uint input_arr_size = inTexture.get_array_size();
    
    uint weithTo = gid.z * kernelHXW * input_arr_size * 4;
    
    float4 output = biase[gid.z];
    
    ushort dilation_y = param.dilationY;
    float4 input[5];
    
    for (uint i = 0; i < input_arr_size; ++i) {
        input[0] = inTexture.sample(sample, float2(posInInput.x, posInInput.y - 2 * dilation_y), i);
        
        input[1] = inTexture.sample(sample, float2(posInInput.x, posInInput.y - dilation_y), i);
        
        input[2] = inTexture.sample(sample, float2(posInInput.x, posInInput.y), i);
        
        input[3] = inTexture.sample(sample, float2(posInInput.x, posInInput.y + dilation_y), i);
        
        input[4] = inTexture.sample(sample, float2(posInInput.x, posInInput.y + 2 * dilation_y), i);
        
        for (int j = 0; j < 5; ++j) {
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
    float4 relu = fmax(output, 0.0);
    outTexture.write(relu, gid.xy, gid.z);
}

kernel void conv_add_relu_1x5(texture2d_array<float, access::sample> inTexture [[texture(0)]],
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
    const ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);
    
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    
    const uint kernelHXW = 5;
    
    uint input_arr_size = inTexture.get_array_size();
    
    uint weithTo = gid.z * kernelHXW * input_arr_size * 4;
    
    float4 output = biase[gid.z];
    
    ushort dilation_x = param.dilationX;
    float4 input[5];
    
    for (uint i = 0; i < input_arr_size; ++i) {
        input[0] = inTexture.sample(sample, float2(posInInput.x - 2 * dilation_x, posInInput.y), i);
        
        input[1] = inTexture.sample(sample, float2(posInInput.x - dilation_x, posInInput.y), i);
        
        input[2] = inTexture.sample(sample, float2(posInInput.x, posInInput.y), i);
        
        input[3] = inTexture.sample(sample, float2(posInInput.x + dilation_x, posInInput.y), i);
        
        input[4] = inTexture.sample(sample, float2(posInInput.x + 2 * dilation_x, posInInput.y), i);
        
        for (int j = 0; j < 5; ++j) {
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
    float4 relu = fmax(output, 0.0);
    outTexture.write(relu, gid.xy, gid.z);
}

kernel void depthwise_conv_add_relu_3x3(texture2d_array<float, access::sample> inTexture [[texture(0)]],
                                   texture2d_array<float, access::write> outTexture [[texture(1)]],
                                   constant MetalConvParam &param [[buffer(0)]],
                                   const device float *weights [[buffer(1)]],
                                   const device float4 *biase [[buffer(2)]],
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
    float4 output = biase[gid.z];
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
    float4 relu = fmax(output, 0.0);
    outTexture.write(relu, gid.xy, gid.z);
}

#pragma mark - half

kernel void conv_add_relu_1x1_half(texture2d_array<half, access::sample> inTexture [[texture(0)]],
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
    
    float4 output = float4(biase[gid.z]);
    
    float4 input;
    for (uint i = 0; i < input_arr_size; ++i) {
        input = float4(inTexture.sample(sample, float2(posInInput.x, posInInput.y), i));
        float4 weight_x = float4(weights[weithTo + 0 * kernelHXW * input_arr_size  + i]);
        output.x += dot(input, weight_x);
        
        float4 weight_y = float4(weights[weithTo + 1 * kernelHXW * input_arr_size  + i]);
        output.y += dot(input, weight_y);
        
        float4 weight_z = float4(weights[weithTo + 2 * kernelHXW * input_arr_size  + i]);
        output.z += dot(input, weight_z);
        
        float4 weight_w = float4(weights[weithTo + 3 * kernelHXW * input_arr_size + i]);
        output.w += dot(input, weight_w);
    }
    float4 relu = fmax(output, 0.0);
    outTexture.write(half4(relu), gid.xy, gid.z);
}

kernel void conv_add_relu_3x3_half(texture2d_array<half, access::sample> inTexture [[texture(0)]],
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
    const ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);
    
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    const uint kernelHXW = 9;
    uint input_arr_size = inTexture.get_array_size();
    uint weithTo = gid.z * kernelHXW * input_arr_size * 4;
    
    float4 output = float4(biase[gid.z]);
    
    ushort dilation_x = param.dilationX;
    ushort dilation_y = param.dilationY;
    
    half4 input[9];
    for (uint i = 0; i < input_arr_size; ++i) {
        input[0] = inTexture.sample(sample, float2(posInInput.x - dilation_x,    posInInput.y - dilation_y), i);
        input[1] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y - dilation_y), i);
        input[2] = inTexture.sample(sample, float2(posInInput.x + dilation_x,    posInInput.y - dilation_y), i);
        input[3] = inTexture.sample(sample, float2(posInInput.x - dilation_x,    posInInput.y), i);
        input[4] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y), i);
        input[5] = inTexture.sample(sample, float2(posInInput.x + dilation_x,    posInInput.y), i);
        input[6] = inTexture.sample(sample, float2(posInInput.x - dilation_x,    posInInput.y + dilation_y), i);
        input[7] = inTexture.sample(sample, float2(posInInput.x,        posInInput.y + dilation_y), i);
        input[8] = inTexture.sample(sample, float2(posInInput.x + dilation_x,    posInInput.y + dilation_y), i);
        for (int j = 0; j < 9; ++j) {
            half4 weight_x = weights[weithTo + 0 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.x += dot(float4(input[j]), float4(weight_x));
            
            half4 weight_y = weights[weithTo + 1 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.y += dot(float4(input[j]), float4(weight_y));
            
            half4 weight_z = weights[weithTo + 2 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.z += dot(float4(input[j]), float4(weight_z));
            
            half4 weight_w = weights[weithTo + 3 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.w += dot(float4(input[j]), float4(weight_w));
        }
    }
    float4 relu = fmax(output, 0.0);
    outTexture.write(half4(relu), gid.xy, gid.z);
}

kernel void depthwise_conv_add_relu_3x3_half(texture2d_array<half, access::sample> inTexture [[texture(0)]],
                                        texture2d_array<half, access::write> outTexture [[texture(1)]],
                                        constant MetalConvParam &param [[buffer(0)]],
                                        const device half *weights [[buffer(1)]],
                                        const device half4 *biase [[buffer(2)]],
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
    float4 output = float4(biase[gid.z]);
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
        output.x += float(input.x) * float(weights[weithTo + 0 * kernelHXW + j]);
        output.y += float(input.y) * float(weights[weithTo + 1 * kernelHXW + j]);
        output.z += float(input.z) * float(weights[weithTo + 2 * kernelHXW + j]);
        output.w += float(input.w) * float(weights[weithTo + 3 * kernelHXW + j]);
    }
    output = fmax(output, 0.0);
    outTexture.write(half4(output), gid.xy, gid.z);
}

kernel void depthwise_conv_add_relu_3x3_half_winograd(texture2d_array<half, access::sample> inTexture [[texture(0)]],
                                             texture2d_array<half, access::write> outTexture [[texture(1)]],
                                             constant MetalConvParam &param [[buffer(0)]],
                                             const device half *weights [[buffer(1)]],
                                             const device half4 *biase [[buffer(2)]],
                                             uint3 gid [[thread_position_in_grid]]) {
    uint ow = outTexture.get_width();
    uint oh = outTexture.get_height();
    if (gid.x >= ow || gid.y >= oh) {
        return;
    }
    
    uint tx = (gid.x / 2) * 2;
    uint ty = (gid.y / 2) * 2;
    uint tc = (gid.x % 2) * 2 + gid.y % 2;
    
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    half4 inputs[16];
    inputs[0] = inTexture.sample(sample, float2(tx - 1, ty - 1), tc);
    inputs[1] = inTexture.sample(sample, float2(tx, ty - 1), tc);
    inputs[2] = inTexture.sample(sample, float2(tx + 1, ty - 1), tc);
    inputs[3] = inTexture.sample(sample, float2(tx + 2, ty - 1), tc);
    
    inputs[4] = inTexture.sample(sample, float2(tx - 1, ty), tc);
    inputs[5] = inTexture.sample(sample, float2(tx, ty), tc);
    inputs[6] = inTexture.sample(sample, float2(tx + 1, ty), tc);
    inputs[7] = inTexture.sample(sample, float2(tx + 2, ty), tc);
    
    inputs[8] = inTexture.sample(sample, float2(tx - 1, ty + 1), tc);
    inputs[9] = inTexture.sample(sample, float2(tx, ty + 1), tc);
    inputs[10] = inTexture.sample(sample, float2(tx + 1, ty + 1), tc);
    inputs[11] = inTexture.sample(sample, float2(tx + 2, ty + 1), tc);
    
    inputs[12] = inTexture.sample(sample, float2(tx - 1, ty + 2), tc);
    inputs[13] = inTexture.sample(sample, float2(tx, ty + 2), tc);
    inputs[14] = inTexture.sample(sample, float2(tx + 1, ty + 2), tc);
    inputs[15] = inTexture.sample(sample, float2(tx + 2, ty + 2), tc);
    
    half4 base = biase[tc];
    half4 res[4] = {base, base, base, base};
    
    half f[3][3];
    const uint kernelHXW = 9;
    uint weightTo = tc * kernelHXW * 4;
    
    for (int c = 0; c < 4; ++c) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                f[i][j] = weights[weightTo++];
            }
        }
        half I[16];
        for (int i = 0; i < 16; ++i) {
            I[i] = inputs[i][c];
        }
        half B[16];
        half tmp1 = I[2] - I[10];
        half tmp2 = I[1] - I[9];
        B[0] = I[0] - I[8] - tmp1;
        B[1] = tmp1 + tmp2;
        B[2] = tmp1 - tmp2;
        B[3] = I[3] - I[11] - tmp2;
        tmp1 = I[6] + I[10];
        tmp2 = I[5] + I[9];
        B[4] = I[4] + I[8] - tmp1;
        B[5] = tmp1 + tmp2;
        B[6] = tmp1 - tmp2;
        B[7] = I[7] + I[11] - tmp2;
        tmp1 = I[6] - I[10];
        tmp2 = I[5] - I[9];
        B[8] = -I[4] + I[8] + tmp1;
        B[9] = -tmp1 - tmp2;
        B[10] = tmp2 - tmp1;
        B[11] = tmp2 - I[7] + I[11];
        tmp1 = I[6] - I[14];
        tmp2 = I[5] - I[13];
        B[12] = -I[4] + I[12] + tmp1;
        B[13] = -tmp1 - tmp2;
        B[14] = tmp2 - tmp1;
        B[15] = tmp2 - I[7] + I[15];
        half G[16];
        G[0] = f[0][0];
        G[1] = 0.5 * f[0][0] + 0.5 * f[0][1] + 0.5 * f[0][2];
        G[2] = 0.5 * f[0][0] - 0.5 * f[0][1] + 0.5 * f[0][2];
        G[3] = f[0][2];
        G[4] = 0.5 * f[0][0] + 0.5 * f[1][0] + 0.5 * f[2][0];
        G[5] = 0.25 * f[0][0] + 0.25 * f[0][1] + 0.25 * f[0][2] + 0.25 * f[1][0] + 0.25 * f[1][1] + 0.25 * f[1][2] + 0.25 * f[2][0] + 0.25 * f[2][1] + 0.25 * f[2][2];
        G[6] = 0.25 * f[0][0] - 0.25 * f[0][1] + 0.25 * f[0][2] + 0.25 * f[1][0] - 0.25 * f[1][1] + 0.25 * f[1][2] + 0.25 * f[2][0] - 0.25 * f[2][1] + 0.25 * f[2][2];
        G[7] = 0.5 * f[0][2] + 0.5 * f[1][2] + 0.5 * f[2][2];
        G[8] = 0.5 * f[0][0] - 0.5 * f[1][0] + 0.5 * f[2][0];
        G[9] = 0.25 * f[0][0] + 0.25 * f[0][1] + 0.25 * f[0][2] - 0.25 * f[1][0] - 0.25 * f[1][1] - 0.25 * f[1][2] + 0.25 * f[2][0] + 0.25 * f[2][1] + 0.25 * f[2][2];
        G[10] = 0.25 * f[0][0] - 0.25 * f[0][1] + 0.25 * f[0][2] - 0.25 * f[1][0] + 0.25 * f[1][1] - 0.25 * f[1][2] + 0.25 * f[2][0] - 0.25 * f[2][1] + 0.25 * f[2][2];
        G[11] = 0.5 * f[0][2] - 0.5 * f[1][2] + 0.5 * f[2][2];
        G[12] = f[2][0];
        G[13] = 0.5 * f[2][0] + 0.5 * f[2][1] + 0.5 * f[2][2];
        G[14] = 0.5 * f[2][0] - 0.5 * f[2][1] + 0.5 * f[2][2];
        G[15] = f[2][2];
        half T[16];
        for (int ii = 0; ii < 16; ++ii) {
            T[ii] = B[ii] * G[ii];
        }
        tmp1 = T[1] + T[5] + T[9];
        tmp2 = T[2] + T[6] + T[10];
        res[0][c] += T[0] + T[4] + T[8] + tmp1 + tmp2;
        res[1][c] += T[3] + T[7] + T[11] + tmp1 - tmp2;
        tmp1 = T[5] - T[9] + T[13];
        tmp2 = T[6] - T[10] + T[14];
        res[2][c] += T[4] - T[8] + T[12] + tmp1 + tmp2;
        res[3][c] += T[7] - T[11] + T[15] + tmp1 - tmp2;
    }
    
    outTexture.write(fmax(res[0], 0.0), uint2(tx, ty), tc);
    outTexture.write(fmax(res[1], 0.0), uint2(tx + 1, ty), tc);
    outTexture.write(fmax(res[2], 0.0), uint2(tx, ty + 1), tc);
    outTexture.write(fmax(res[3], 0.0), uint2(tx + 1, ty + 1), tc);
}

//kernel void depthwise_conv_add_relu_3x3_half_winograd_naive(texture2d_array<half, access::sample> inTexture [[texture(0)]],
//                                                      texture2d_array<half, access::write> outTexture [[texture(1)]],
//                                                      constant MetalConvParam &param [[buffer(0)]],
//                                                      const device half *weights [[buffer(1)]],
//                                                      const device half4 *biase [[buffer(2)]],
//                                                      uint3 gid [[thread_position_in_grid]]) {
//    uint ow = outTexture.get_width();
//    uint oh = outTexture.get_height();
//    if (gid.x >= ow || gid.y >= oh) {
//        return;
//    }
//
//    uint tx = (gid.x / 2) * 2;
//    uint ty = (gid.y / 2) * 2;
//    uint tc = (gid.x % 2) * 2 + gid.y % 2;
//
//    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
//    half4 inputs[4][4];
//    inputs[0][0] = inTexture.sample(sample, float2(tx - 1, ty - 1), tc);
//    inputs[0][1] = inTexture.sample(sample, float2(tx, ty - 1), tc);
//    inputs[0][2] = inTexture.sample(sample, float2(tx + 1, ty - 1), tc);
//    inputs[0][3] = inTexture.sample(sample, float2(tx + 2, ty - 1), tc);
//
//    inputs[1][0] = inTexture.sample(sample, float2(tx - 1, ty), tc);
//    inputs[1][1] = inTexture.sample(sample, float2(tx, ty), tc);
//    inputs[1][2] = inTexture.sample(sample, float2(tx + 1, ty), tc);
//    inputs[1][3] = inTexture.sample(sample, float2(tx + 2, ty), tc);
//
//    inputs[2][0] = inTexture.sample(sample, float2(tx - 1, ty + 1), tc);
//    inputs[2][1] = inTexture.sample(sample, float2(tx, ty + 1), tc);
//    inputs[2][2] = inTexture.sample(sample, float2(tx + 1, ty + 1), tc);
//    inputs[2][3] = inTexture.sample(sample, float2(tx + 2, ty + 1), tc);
//
//    inputs[3][0] = inTexture.sample(sample, float2(tx - 1, ty + 2), tc);
//    inputs[3][1] = inTexture.sample(sample, float2(tx, ty + 2), tc);
//    inputs[3][2] = inTexture.sample(sample, float2(tx + 1, ty + 2), tc);
//    inputs[3][3] = inTexture.sample(sample, float2(tx + 2, ty + 2), tc);
//
//    const uint kernelHXW = 9;
//    uint weightTo = tc * kernelHXW * 4;
//
//    half f[3][3];
//
//    half4 base = biase[tc];
//    half4 res[2][2];
//    res[0][0] = base;
//    res[0][1] = base;
//    res[1][0] = base;
//    res[1][1] = base;
//
//    for (int c = 0; c < 4; ++c) {
//        for (int i = 0; i < 3; ++i) {
//            for (int j = 0; j < 3; ++j) {
//                f[i][j] = weights[weightTo++];
//            }
//        }
//        half I[4][4];
//        for (int ii = 0; ii < 4; ++ii) {
//            for (int jj = 0; jj < 4; ++jj) {
//                I[ii][jj] = inputs[ii][jj][c];
//            }
//        }
//        half B[4][4];
//        B[0][0] = I[0][0] - I[0][2] - I[2][0] + I[2][2];
//        B[0][1] = I[0][1] + I[0][2] - I[2][1] - I[2][2];
//        B[0][2] = -I[0][1] + I[0][2] + I[2][1] - I[2][2];
//        B[0][3] = -I[0][1] + I[0][3] + I[2][1] - I[2][3];
//        B[1][0] = I[1][0] - I[1][2] + I[2][0] - I[2][2];
//        B[1][1] = I[1][1] + I[1][2] + I[2][1] + I[2][2];
//        B[1][2] = -I[1][1] + I[1][2] - I[2][1] + I[2][2];
//        B[1][3] = -I[1][1] + I[1][3] - I[2][1] + I[2][3];
//        B[2][0] = -I[1][0] + I[1][2] + I[2][0] - I[2][2];
//        B[2][1] = -I[1][1] - I[1][2] + I[2][1] + I[2][2];
//        B[2][2] = I[1][1] - I[1][2] - I[2][1] + I[2][2];
//        B[2][3] = I[1][1] - I[1][3] - I[2][1] + I[2][3];
//        B[3][0] = -I[1][0] + I[1][2] + I[3][0] - I[3][2];
//        B[3][1] = -I[1][1] - I[1][2] + I[3][1] + I[3][2];
//        B[3][2] = I[1][1] - I[1][2] - I[3][1] + I[3][2];
//        B[3][3] = I[1][1] - I[1][3] - I[3][1] + I[3][3];
//        half G[4][4];
//        G[0][0] = f[0][0];
//        G[0][1] = 0.5 * f[0][0] + 0.5 * f[0][1] + 0.5 * f[0][2];
//        G[0][2] = 0.5 * f[0][0] - 0.5 * f[0][1] + 0.5 * f[0][2];
//        G[0][3] = f[0][2];
//        G[1][0] = 0.5 * f[0][0] + 0.5 * f[1][0] + 0.5 * f[2][0];
//        G[1][1] = 0.25 * f[0][0] + 0.25 * f[0][1] + 0.25 * f[0][2] + 0.25 * f[1][0] + 0.25 * f[1][1] + 0.25 * f[1][2] + 0.25 * f[2][0] + 0.25 * f[2][1] + 0.25 * f[2][2];
//        G[1][2] = 0.25 * f[0][0] - 0.25 * f[0][1] + 0.25 * f[0][2] + 0.25 * f[1][0] - 0.25 * f[1][1] + 0.25 * f[1][2] + 0.25 * f[2][0] - 0.25 * f[2][1] + 0.25 * f[2][2];
//        G[1][3] = 0.5 * f[0][2] + 0.5 * f[1][2] + 0.5 * f[2][2];
//        G[2][0] = 0.5 * f[0][0] - 0.5 * f[1][0] + 0.5 * f[2][0];
//        G[2][1] = 0.25 * f[0][0] + 0.25 * f[0][1] + 0.25 * f[0][2] - 0.25 * f[1][0] - 0.25 * f[1][1] - 0.25 * f[1][2] + 0.25 * f[2][0] + 0.25 * f[2][1] + 0.25 * f[2][2];
//        G[2][2] = 0.25 * f[0][0] - 0.25 * f[0][1] + 0.25 * f[0][2] - 0.25 * f[1][0] + 0.25 * f[1][1] - 0.25 * f[1][2] + 0.25 * f[2][0] - 0.25 * f[2][1] + 0.25 * f[2][2];
//        G[2][3] = 0.5 * f[0][2] - 0.5 * f[1][2] + 0.5 * f[2][2];
//        G[3][0] = f[2][0];
//        G[3][1] = 0.5 * f[2][0] + 0.5 * f[2][1] + 0.5 * f[2][2];
//        G[3][2] = 0.5 * f[2][0] - 0.5 * f[2][1] + 0.5 * f[2][2];
//        G[3][3] = f[2][2];
//        half T[4][4];
//        for (int ii = 0; ii < 4; ++ii) {
//            for (int jj = 0; jj < 4; ++jj) {
//                T[ii][jj] = B[ii][jj] * G[ii][jj];
//            }
//        }
//        half A[2][2];
//        A[0][0] = T[0][0] + T[0][1] + T[0][2] + T[1][0] + T[1][1] + T[1][2] + T[2][0] + T[2][1] + T[2][2];
//        A[0][1] = T[0][1] - T[0][2] + T[0][3] + T[1][1] - T[1][2] + T[1][3] + T[2][1] - T[2][2] + T[2][3];
//        A[1][0] = T[1][0] + T[1][1] + T[1][2] - T[2][0] - T[2][1] - T[2][2] + T[3][0] + T[3][1] + T[3][2];
//        A[1][1] = T[1][1] - T[1][2] + T[1][3] - T[2][1] + T[2][2] - T[2][3] + T[3][1] - T[3][2] + T[3][3];
//        for (int i = 0; i < 2; ++i) {
//            for (int j = 0; j < 2; ++j) {
//                res[i][j][c] += A[i][j];
//            }
//        }
//    }
//
//    for (int i = 0; i < 2; ++i) {
//        for (int j = 0; j < 2; ++j) {
//            half4 output = fmax(res[i][j], 0.0);
//            outTexture.write(output, uint2(tx + j, ty + i), tc);
//        }
//    }
//}

kernel void conv_add_relu_5x1_half(texture2d_array<half, access::sample> inTexture [[texture(0)]],
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
    const ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);
    
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    
    const uint kernelHXW = 5;
    
    uint input_arr_size = inTexture.get_array_size();
    
    uint weithTo = gid.z * kernelHXW * input_arr_size * 4;
    
    float4 output = float4(biase[gid.z]);
    
    ushort dilation_y = param.dilationY;
    half4 input[5];
    
    for (uint i = 0; i < input_arr_size; ++i) {
        input[0] = inTexture.sample(sample, float2(posInInput.x, posInInput.y - 2 * dilation_y), i);
        
        input[1] = inTexture.sample(sample, float2(posInInput.x, posInInput.y - dilation_y), i);
        
        input[2] = inTexture.sample(sample, float2(posInInput.x, posInInput.y), i);
        
        input[3] = inTexture.sample(sample, float2(posInInput.x, posInInput.y + dilation_y), i);
        
        input[4] = inTexture.sample(sample, float2(posInInput.x, posInInput.y + 2 * dilation_y), i);
        
        for (int j = 0; j < 5; ++j) {
            half4 weight_x = weights[weithTo + 0 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.x += dot(float4(input[j]), float4(weight_x));
            
            half4 weight_y = weights[weithTo + 1 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.y += dot(float4(input[j]), float4(weight_y));
            
            half4 weight_z = weights[weithTo + 2 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.z += dot(float4(input[j]), float4(weight_z));
            
            half4 weight_w = weights[weithTo + 3 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.w += dot(float4(input[j]), float4(weight_w));
        }
    }
    float4 relu = fmax(output, 0.0);
    outTexture.write(half4(relu), gid.xy, gid.z);
}

kernel void conv_add_relu_1x5_half(texture2d_array<half, access::sample> inTexture [[texture(0)]],
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
    const ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);
    
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    
    const uint kernelHXW = 5;
    
    uint input_arr_size = inTexture.get_array_size();
    
    uint weithTo = gid.z * kernelHXW * input_arr_size * 4;
    
    float4 output = float4(biase[gid.z]);
    
    ushort dilation_x = param.dilationX;
    half4 input[5];
    
    for (uint i = 0; i < input_arr_size; ++i) {
        input[0] = inTexture.sample(sample, float2(posInInput.x - 2 * dilation_x, posInInput.y), i);
        
        input[1] = inTexture.sample(sample, float2(posInInput.x - dilation_x, posInInput.y), i);
        
        input[2] = inTexture.sample(sample, float2(posInInput.x, posInInput.y), i);
        
        input[3] = inTexture.sample(sample, float2(posInInput.x + dilation_x, posInInput.y), i);
        
        input[4] = inTexture.sample(sample, float2(posInInput.x + 2 * dilation_x, posInInput.y), i);
        
        for (int j = 0; j < 5; ++j) {
            half4 weight_x = weights[weithTo + 0 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.x += dot(float4(input[j]), float4(weight_x));
            
            half4 weight_y = weights[weithTo + 1 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.y += dot(float4(input[j]), float4(weight_y));
            
            half4 weight_z = weights[weithTo + 2 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.z += dot(float4(input[j]), float4(weight_z));
            
            half4 weight_w = weights[weithTo + 3 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.w += dot(float4(input[j]), float4(weight_w));
        }
    }
    float4 relu = fmax(output, 0.0);
    outTexture.write(half4(relu), gid.xy, gid.z);
}
