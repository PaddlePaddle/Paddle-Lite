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

half4 getBiasHalf(uint3 gid, constant ElementwiseAddParam &addParam, texture2d_array<half, access::sample> biasTexture) {
    half4 output;
    if (addParam.fast == 1) {
        output = biasTexture.read(gid.xy, gid.z);
    } else if (addParam.addByChannel == 1) {
        output = biasTexture.read(uint2(0, 0), gid.z);
    } else {
        int32_t x_xyzn[4] = {int32_t(gid.x), int32_t(gid.y), int32_t(gid.z), 0}, x_abcd[4], t_abcd[4];
        int32_t y_abcd[4] = {0, 0, 0, 0}, y_xyzn[4];
        int32_t xtrans[4] = {addParam.xtrans[0], addParam.xtrans[1], addParam.xtrans[2], addParam.xtrans[3]};
        int32_t ytrans[4] = {addParam.ytrans[0], addParam.ytrans[1], addParam.ytrans[2], addParam.ytrans[3]};
        int32_t yshift = 4 - addParam.ylen - addParam.axis;
        for (int n = 0; n < 4; n++) {
            x_xyzn[3] = n;
            xyzn2abcd(addParam.xdim[3], x_xyzn, x_abcd);
            invtrans(xtrans, x_abcd, t_abcd);
            for (int k = addParam.axis; k < (addParam.axis + addParam.ylen); k++) {
                y_abcd[yshift+k] = t_abcd[k];
            }
            trans(ytrans, y_abcd, t_abcd);
            abcd2xyzn(addParam.ydim[3], t_abcd, y_xyzn);
            output[n] = biasTexture.read(uint2(y_xyzn[0], y_xyzn[1]), y_xyzn[2])[y_xyzn[3]];
        }
    }
    return output;
}

float4 getBias(uint3 gid, constant ElementwiseAddParam &addParam, texture2d_array<float, access::sample> biasTexture) {
    float4 output;
    if (addParam.fast == 1) {
        output = float4(biasTexture.read(gid.xy, gid.z));
    } else if (addParam.addByChannel == 1) {
        output = float4(biasTexture.read(uint2(0, 0), gid.z));
    } else {
        int32_t x_xyzn[4] = {int32_t(gid.x), int32_t(gid.y), int32_t(gid.z), 0}, x_abcd[4], t_abcd[4];
        int32_t y_abcd[4] = {0, 0, 0, 0}, y_xyzn[4];
        int32_t xtrans[4] = {addParam.xtrans[0], addParam.xtrans[1], addParam.xtrans[2], addParam.xtrans[3]};
        int32_t ytrans[4] = {addParam.ytrans[0], addParam.ytrans[1], addParam.ytrans[2], addParam.ytrans[3]};
        int32_t yshift = 4 - addParam.ylen - addParam.axis;
        for (int n = 0; n < 4; n++) {
            x_xyzn[3] = n;
            xyzn2abcd(addParam.xdim[3], x_xyzn, x_abcd);
            invtrans(xtrans, x_abcd, t_abcd);
            for (int k = addParam.axis; k < (addParam.axis + addParam.ylen); k++) {
                y_abcd[yshift+k] = t_abcd[k];
            }
            trans(ytrans, y_abcd, t_abcd);
            abcd2xyzn(addParam.ydim[3], t_abcd, y_xyzn);
            output[n] = biasTexture.read(uint2(y_xyzn[0], y_xyzn[1]), y_xyzn[2])[y_xyzn[3]];
        }
    }
    return output;
}

#pragma mark - convAdd
kernel void conv_add_relu_1x1(texture2d_array<float, access::sample> inTexture [[texture(0)]],
                         texture2d_array<float, access::sample> biasTexture [[texture(1)]],
                         texture2d_array<float, access::write> outTexture [[texture(2)]],
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
    
    float4 output = float4(0.0, 0.0, 0.0, 0.0);
    if (param.hasAddOp) {
        constant ElementwiseAddParam &addParam = param.addParam;
        output = getBias(gid, addParam, biasTexture);
    }
    
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
    float4 relu = param.hasReluOp == 1 ? fmax(output, 0.0) : output;
    outTexture.write(relu, gid.xy, gid.z);
}

kernel void conv_add_relu_3x3(texture2d_array<float, access::sample> inTexture [[texture(0)]],
                         texture2d_array<float, access::sample> biasTexture [[texture(1)]],
                         texture2d_array<float, access::write> outTexture [[texture(2)]],
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
    
    float4 output = float4(0.0, 0.0, 0.0, 0.0);
    if (param.hasAddOp) {
        constant ElementwiseAddParam &addParam = param.addParam;
        output = getBias(gid, addParam, biasTexture);
    }
    
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
    float4 relu = param.hasReluOp == 1 ? fmax(output, 0.0) : output;
    outTexture.write(relu, gid.xy, gid.z);
}

kernel void group_conv_add_relu_3x3(texture2d_array<float, access::sample> inTexture [[texture(0)]],
                              texture2d_array<float, access::sample> biasTexture [[texture(1)]],
                              texture2d_array<float, access::write> outTexture [[texture(2)]],
                              constant MetalConvParam &param [[buffer(0)]],
                              const device float *weights [[buffer(1)]],
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
    
    float4 output = float4(0.0, 0.0, 0.0, 0.0);
    if (param.hasAddOp) {
        constant ElementwiseAddParam &addParam = param.addParam;
        output = getBias(gid, addParam, biasTexture);
    }
    
    ushort dilation_x = param.dilationX;
    ushort dilation_y = param.dilationY;
    
    float input[9];
    
    uint iC = param.iC, fC = param.fC, oC = param.oC;
    uint filter_array_size = (fC + 3) / 4;
    
    for (uint c = 0; c < 4; ++c) {
        uint output_depth = gid.z * 4 + c, output_c = output_depth % oC, output_n = output_depth / oC;
        for (uint i = 0; i < fC; ++i) {
            uint input_depth = output_n * iC + output_c * fC + i;
            uint input_array_index = input_depth / 4;
            uint input_array_item_index = input_depth % 4;
            input[0] = inTexture.sample(sample, float2(posInInput.x - dilation_x, posInInput.y - dilation_y), input_array_index)[input_array_item_index];
            input[1] = inTexture.sample(sample, float2(posInInput.x, posInInput.y - dilation_y), input_array_index)[input_array_item_index];
            input[2] = inTexture.sample(sample, float2(posInInput.x + dilation_x, posInInput.y - dilation_y), input_array_index)[input_array_item_index];
            input[3] = inTexture.sample(sample, float2(posInInput.x - dilation_x, posInInput.y), input_array_index)[input_array_item_index];
            input[4] = inTexture.sample(sample, float2(posInInput.x, posInInput.y), input_array_index)[input_array_item_index];
            input[5] = inTexture.sample(sample, float2(posInInput.x + dilation_x, posInInput.y), input_array_index)[input_array_item_index];
            input[6] = inTexture.sample(sample, float2(posInInput.x - dilation_x, posInInput.y + dilation_y), input_array_index)[input_array_item_index];
            input[7] = inTexture.sample(sample, float2(posInInput.x, posInInput.y + dilation_y), input_array_index)[input_array_item_index];
            input[8] = inTexture.sample(sample, float2(posInInput.x + dilation_x, posInInput.y + dilation_y), input_array_index)[input_array_item_index];
            for (int j = 0; j < 9; ++j) {
                float weight = weights[(output_c * kernelHXW + j) * filter_array_size * 4 + i];
                output[c] += input[j] * weight;
            }
        }
    }
    
    float4 relu = param.hasReluOp == 1 ? fmax(output, 0.0) : output;
    outTexture.write(relu, gid.xy, gid.z);
}

kernel void conv_add_relu_5x1(texture2d_array<float, access::sample> inTexture [[texture(0)]],
                         texture2d_array<float, access::sample> biasTexture [[texture(1)]],
                         texture2d_array<float, access::write> outTexture [[texture(2)]],
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
    
    const uint kernelHXW = 5;
    
    uint input_arr_size = inTexture.get_array_size();
    
    uint weithTo = gid.z * kernelHXW * input_arr_size * 4;
    
    float4 output = float4(0.0, 0.0, 0.0, 0.0);
    if (param.hasAddOp) {
        constant ElementwiseAddParam &addParam = param.addParam;
        output = getBias(gid, addParam, biasTexture);
    }
    
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
    float4 relu = param.hasReluOp == 1 ? fmax(output, 0.0) : output;
    outTexture.write(relu, gid.xy, gid.z);
}

kernel void conv_add_relu_1x5(texture2d_array<float, access::sample> inTexture [[texture(0)]],
                         texture2d_array<float, access::sample> biasTexture [[texture(1)]],
                         texture2d_array<float, access::write> outTexture [[texture(2)]],
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
    
    const uint kernelHXW = 5;
    
    uint input_arr_size = inTexture.get_array_size();
    
    uint weithTo = gid.z * kernelHXW * input_arr_size * 4;
    
    float4 output = float4(0.0, 0.0, 0.0, 0.0);
    if (param.hasAddOp) {
        constant ElementwiseAddParam &addParam = param.addParam;
        output = getBias(gid, addParam, biasTexture);
    }
    
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
    float4 relu = param.hasReluOp == 1 ? fmax(output, 0.0) : output;
    outTexture.write(relu, gid.xy, gid.z);
}

kernel void depthwise_conv_add_relu_3x3(texture2d_array<float, access::sample> inTexture [[texture(0)]],
                                   texture2d_array<float, access::sample> biasTexture [[texture(1)]],
                                   texture2d_array<float, access::write> outTexture [[texture(2)]],
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

    float4 output = float4(0.0, 0.0, 0.0, 0.0);
    if (param.hasAddOp) {
        constant ElementwiseAddParam &addParam = param.addParam;
        output = getBias(gid, addParam, biasTexture);
    }
    
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
    float4 relu = param.hasReluOp == 1 ? fmax(output, 0.0) : output;
    outTexture.write(relu, gid.xy, gid.z);
}

#pragma mark - half

kernel void conv_add_relu_1x1_half(texture2d_array<half, access::sample> inTexture [[texture(0)]],
                              texture2d_array<half, access::sample> biasTexture [[texture(1)]],
                              texture2d_array<half, access::write> outTexture [[texture(2)]],
                              constant MetalConvParam &param [[buffer(0)]],
                              const device half4 *weights [[buffer(1)]],
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
    
    float4 output = float4(0.0, 0.0, 0.0, 0.0);
    if (param.hasAddOp) {
        constant ElementwiseAddParam &addParam = param.addParam;
        output = float4(getBiasHalf(gid, addParam, biasTexture));
    }
    
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
    float4 relu = param.hasReluOp == 1 ? fmax(output, 0.0) : output;
    outTexture.write(half4(relu), gid.xy, gid.z);
}

kernel void conv_add_relu_3x3_half(texture2d_array<half, access::sample> inTexture [[texture(0)]],
                              texture2d_array<half, access::sample> biasTexture [[texture(1)]],
                              texture2d_array<half, access::write> outTexture [[texture(2)]],
                              constant MetalConvParam &param [[buffer(0)]],
                              const device half4 *weights [[buffer(1)]],
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
    
    float4 output = float4(0.0, 0.0, 0.0, 0.0);
    if (param.hasAddOp) {
        constant ElementwiseAddParam &addParam = param.addParam;
        output = float4(getBiasHalf(gid, addParam, biasTexture));
    }

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
    float4 relu = param.hasReluOp == 1 ? fmax(output, 0.0) : output;
    outTexture.write(half4(relu), gid.xy, gid.z);
}

kernel void group_conv_add_relu_3x3_half(texture2d_array<half, access::sample> inTexture [[texture(0)]],
                                    texture2d_array<half, access::sample> biasTexture [[texture(1)]],
                                    texture2d_array<half, access::write> outTexture [[texture(2)]],
                                    constant MetalConvParam &param [[buffer(0)]],
                                    const device half *weights [[buffer(1)]],
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
    
    float4 output = float4(0.0, 0.0, 0.0, 0.0);
    if (param.hasAddOp) {
        constant ElementwiseAddParam &addParam = param.addParam;
        output = float4(getBiasHalf(gid, addParam, biasTexture));
    }
    
    ushort dilation_x = param.dilationX;
    ushort dilation_y = param.dilationY;
    
    half input[9];
    
    uint iC = param.iC, fC = param.fC, oC = param.oC;
    uint filter_array_size = (fC + 3) / 4;
    
    for (uint c = 0; c < 4; ++c) {
        uint output_depth = gid.z * 4 + c, output_c = output_depth % oC, output_n = output_depth / oC;
        for (uint i = 0; i < fC; ++i) {
            uint input_depth = output_n * iC + output_c * fC + i;
            uint input_array_index = input_depth / 4;
            uint input_array_item_index = input_depth % 4;
            input[0] = inTexture.sample(sample, float2(posInInput.x - dilation_x, posInInput.y - dilation_y), input_array_index)[input_array_item_index];
            input[1] = inTexture.sample(sample, float2(posInInput.x, posInInput.y - dilation_y), input_array_index)[input_array_item_index];
            input[2] = inTexture.sample(sample, float2(posInInput.x + dilation_x, posInInput.y - dilation_y), input_array_index)[input_array_item_index];
            input[3] = inTexture.sample(sample, float2(posInInput.x - dilation_x, posInInput.y), input_array_index)[input_array_item_index];
            input[4] = inTexture.sample(sample, float2(posInInput.x, posInInput.y), input_array_index)[input_array_item_index];
            input[5] = inTexture.sample(sample, float2(posInInput.x + dilation_x, posInInput.y), input_array_index)[input_array_item_index];
            input[6] = inTexture.sample(sample, float2(posInInput.x - dilation_x, posInInput.y + dilation_y), input_array_index)[input_array_item_index];
            input[7] = inTexture.sample(sample, float2(posInInput.x, posInInput.y + dilation_y), input_array_index)[input_array_item_index];
            input[8] = inTexture.sample(sample, float2(posInInput.x + dilation_x, posInInput.y + dilation_y), input_array_index)[input_array_item_index];
            for (int j = 0; j < 9; ++j) {
                half weight = weights[(output_c * kernelHXW + j) * filter_array_size * 4 + i];
                output[c] += float(input[j]) * float(weight);
            }
        }
    }
    
    float4 relu = param.hasReluOp == 1 ? fmax(output, 0.0) : output;
    outTexture.write(half4(relu), gid.xy, gid.z);
}

kernel void depthwise_conv_add_relu_3x3_half(texture2d_array<half, access::sample> inTexture [[texture(0)]],
                                        texture2d_array<half, access::sample> biasTexture [[texture(1)]],
                                        texture2d_array<half, access::write> outTexture [[texture(2)]],
                                        constant MetalConvParam &param [[buffer(0)]],
                                        const device half *weights [[buffer(1)]],
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

    float4 output = float4(0.0, 0.0, 0.0, 0.0);
    if (param.hasAddOp) {
        constant ElementwiseAddParam &addParam = param.addParam;
        output = float4(getBiasHalf(gid, addParam, biasTexture));
    }

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

    float4 relu = param.hasReluOp == 1 ? fmax(output, 0.0) : output;
    outTexture.write(half4(relu), gid.xy, gid.z);
}

kernel void depthwise_conv_add_relu_3x3_half_winograd(texture2d_array<half, access::sample> inTexture [[texture(0)]],
                                             texture2d_array<half, access::sample> biasTexture [[texture(1)]],
                                             texture2d_array<half, access::write> outTexture [[texture(2)]],
                                             constant MetalConvParam &param [[buffer(0)]],
                                             const device half4x4 *weights [[buffer(1)]],
                                             uint3 gid [[thread_position_in_grid]]) {
    uint x = gid.x, y = gid.y;
    uint ow = outTexture.get_width();
    if (ow % 2 != 0) {
        ow++;
    }
    uint oh = outTexture.get_height();
    if (oh % 2 != 0) {
        oh++;
    }
    if (x >= ow || y >= oh) {
        return;
    }

    uint tx = (x >> 1) << 1;
    uint ty = (y >> 1) << 1;
    uint tc = ((x % 2) << 1) + y % 2;
    
    int hasComputedC = 4 * tc;
    
    if (hasComputedC >= param.oC) {
        return;
    }

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

    uint weightTo = 4 * tc;
    half4 res[4];

    for (int c = 0; c < 4; ++c) {
        if (hasComputedC + c >= param.oC) {
            break;
        }
        half I[16];
        for (int i = 0; i < 16; ++i) {
            I[i] = inputs[i][c];
        }
        half4x4 f = weights[weightTo + c];
        half B[16];
        half tmp1 = I[2] - I[10];
        half tmp2 = I[9] - I[1];
        B[0] = I[0] - I[8] - tmp1;
        B[1] = tmp1 - tmp2;
        B[2] = tmp1 + tmp2;
        B[3] = I[3] - I[11] + tmp2;
        tmp1 = I[6] + I[10];
        tmp2 = I[5] + I[9];
        B[4] = I[4] + I[8] - tmp1;
        B[5] = tmp1 + tmp2;
        B[6] = tmp1 - tmp2;
        B[7] = I[7] + I[11] - tmp2;
        tmp1 = I[10] - I[6];
        tmp2 = I[5] - I[9];
        B[8] = I[8] - I[4] - tmp1;
        B[9] = tmp1 - tmp2;
        B[10] = tmp1 + tmp2;
        B[11] = tmp2 - I[7] + I[11];
        tmp1 = I[14] - I[6];
        tmp2 = I[5] - I[13];
        B[12] = I[12] - I[4] - tmp1;
        B[13] = tmp1 - tmp2;
        B[14] = tmp1 + tmp2;
        B[15] = tmp2 - I[7] + I[15];
        half T[16];
        T[0] = B[0] * f[0][0];
        T[1] = B[1] * f[0][1];
        T[2] = B[2] * f[0][2];
        T[3] = B[3] * f[0][3];
        T[4] = B[4] * f[1][0];
        T[5] = B[5] * f[1][1];
        T[6] = B[6] * f[1][2];
        T[7] = B[7] * f[1][3];
        T[8] = B[8] * f[2][0];
        T[9] = B[9] * f[2][1];
        T[10] = B[10] * f[2][2];
        T[11] = B[11] * f[2][3];
        T[12] = B[12] * f[3][0];
        T[13] = B[13] * f[3][1];
        T[14] = B[14] * f[3][2];
        T[15] = B[15] * f[3][3];
        tmp1 = T[1] + T[5] + T[9];
        tmp2 = T[2] + T[6] + T[10];
        res[0][c] = T[0] + T[4] + T[8] + tmp1 + tmp2;
        res[1][c] = T[3] + T[7] + T[11] + tmp1 - tmp2;
        tmp1 = T[5] - T[9] + T[13];
        tmp2 = T[6] - T[10] + T[14];
        res[2][c] = T[4] - T[8] + T[12] + tmp1 + tmp2;
        res[3][c] = T[7] - T[11] + T[15] + tmp1 - tmp2;
    }
    
    if (param.hasAddOp == 1) {
        constant ElementwiseAddParam &addParam = param.addParam;
        half4 base = getBiasHalf(uint3(tx, ty, tc), addParam, biasTexture);
        res[0] += base;
        base = getBiasHalf(uint3(tx + 1, ty, tc), addParam, biasTexture);
        res[1] += base;
        base = getBiasHalf(uint3(tx, ty + 1, tc), addParam, biasTexture);
        res[2] += base;
        base = getBiasHalf(uint3(tx + 1, ty + 1, tc), addParam, biasTexture);
        res[3] += base;
    }

    if (param.hasReluOp == 1) {
        outTexture.write(fmax(res[0], 0.0), uint2(tx, ty), tc);
        outTexture.write(fmax(res[1], 0.0), uint2(tx + 1, ty), tc);
        outTexture.write(fmax(res[2], 0.0), uint2(tx, ty + 1), tc);
        outTexture.write(fmax(res[3], 0.0), uint2(tx + 1, ty + 1), tc);
    } else {
        outTexture.write(res[0], uint2(tx, ty), tc);
        outTexture.write(res[1], uint2(tx + 1, ty), tc);
        outTexture.write(res[2], uint2(tx, ty + 1), tc);
        outTexture.write(res[3], uint2(tx + 1, ty + 1), tc);
    }
}

kernel void conv_add_relu_5x1_half(texture2d_array<half, access::sample> inTexture [[texture(0)]],
                              texture2d_array<half, access::sample> biasTexture [[texture(1)]],
                              texture2d_array<half, access::write> outTexture [[texture(2)]],
                              constant MetalConvParam &param [[buffer(0)]],
                              const device half4 *weights [[buffer(1)]],
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
    
    float4 output = float4(0.0, 0.0, 0.0, 0.0);
    if (param.hasAddOp) {
        constant ElementwiseAddParam &addParam = param.addParam;
        output = float4(getBiasHalf(gid, addParam, biasTexture));
    }

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
    float4 relu = param.hasReluOp == 1 ? fmax(output, 0.0) : output;
    outTexture.write(half4(relu), gid.xy, gid.z);
}

kernel void conv_add_relu_1x5_half(texture2d_array<half, access::sample> inTexture [[texture(0)]],
                              texture2d_array<half, access::sample> biasTexture [[texture(1)]],
                              texture2d_array<half, access::write> outTexture [[texture(2)]],
                              constant MetalConvParam &param [[buffer(0)]],
                              const device half4 *weights [[buffer(1)]],
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
    
    float4 output = float4(0.0, 0.0, 0.0, 0.0);
    if (param.hasAddOp) {
        constant ElementwiseAddParam &addParam = param.addParam;
        output = float4(getBiasHalf(gid, addParam, biasTexture));
    }

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
    float4 relu = param.hasReluOp == 1 ? fmax(output, 0.0) : output;
    outTexture.write(half4(relu), gid.xy, gid.z);
}
