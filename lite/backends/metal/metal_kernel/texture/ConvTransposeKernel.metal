/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

inline ftype4 get_bias(uint3 gid,
    constant ElementwiseAddParam& addParam,
    texture2d_array<ftype, access::sample> biasTexture) {
    ftype4 output = ftype4(0.0);
    if (addParam.fast == 1) {
        output = biasTexture.read(gid.xy, gid.z);
    } else if (addParam.addByChannel == 1) {
        output = biasTexture.read(uint2(0, 0), gid.z);
    } else {
    }
    return output;
}

kernel void conv_transpose2x2_stride2(
    texture2d_array<ftype, access::sample> inTexture[[texture(0)]],
    texture2d_array<ftype, access::sample> biasTexture[[texture(1)]],
    texture2d_array<ftype, access::write> outTexture[[texture(2)]],
    constant MetalConvTransposeParam& param[[buffer(0)]],
    const device ftype4* weights[[buffer(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }

    int input_array_size = inTexture.get_array_size();
    int kernel_index_x = gid.x % 2;
    int kernel_index_y = gid.y % 2;
    int kernel_index = kernel_index_y * 2 + kernel_index_x;
    int kernel_to = gid.z * input_array_size * 4 * 4 + (kernel_index * input_array_size);
    int input_x = gid.x / 2;
    int input_y = gid.y / 2;

    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    ftype4 output = ftype4(0.0);
    if (param.hasAddOp) {
        output = get_bias(gid, param.addParam, biasTexture);
    }

    for (int i = 0; i < input_array_size; ++i) {
        ftype4 input = inTexture.sample(sample, float2(input_x, input_y), i);

        ftype4 kernel_slice0 = weights[kernel_to + input_array_size * 4 * 0 + i];
        ftype4 kernel_slice1 = weights[kernel_to + input_array_size * 4 * 1 + i];
        ftype4 kernel_slice2 = weights[kernel_to + input_array_size * 4 * 2 + i];
        ftype4 kernel_slice3 = weights[kernel_to + input_array_size * 4 * 3 + i];

        output.x += dot(input, kernel_slice0);

        output.y += dot(input, kernel_slice1);

        output.z += dot(input, kernel_slice2);

        output.w += dot(input, kernel_slice3);
    }

    outTexture.write(output, gid.xy, gid.z);
}

// kernel void conv_transpose(texture2d_array<float, access::sample> inTexture
// [[texture(0)]],
//                           texture2d_array<float, access::write> outTexture
//                           [[texture(1)]], constant MetalConvTransposeParam
//                           &param [[buffer(0)]], const device float4 *weights
//                           [[buffer(1)]], uint3 gid
//                           [[thread_position_in_grid]]){
//  if (gid.x >= outTexture.get_width() ||
//      gid.y >= outTexture.get_height() ||
//      gid.z >= outTexture.get_array_size()) {
//    return;
//  }
//
//  int input_array_size = inTexture.get_array_size();
//
//  uint kernel_one_output_slice = input_array_size * param.kernelW *
//  param.kernelH;
//
//  uint kernel_stride_z = gid.z * 4 * (kernel_one_output_slice);
//
//  constexpr sampler sample(coord::pixel, filter::nearest,
//  address::clamp_to_zero);
//
//  float4 output;
//
//  for (int w = 0; w < param.kernelW; ++w) {
//    int top = gid.x - w * param.dilationX + param.paddingX;
//    int input_x = top / param.strideX;
//    if (top < 0 || input_x >= int(inTexture.get_width())) {
//      continue;
//    }
//
//    for (int h = 0; h < param.kernelH; ++h) {
//      int top_y = gid.y - h * param.dilationY + param.paddingY;
//      int input_y = top_y / param.strideY;
//      if (top_y < 0 || input_y >= int(inTexture.get_height())) {
//        continue;
//      }
//
//      uint kernel_index = (w * param.kernelH + h) *
//      inTexture.get_array_size();
//
//      for (int slice = 0; slice < input_array_size; ++slice) {
//
//        float4 input;
//        float4 kernel_slice = weights[kernel_stride_z + 0 *
//        kernel_one_output_slice + kernel_index + slice]; float4 kernel_slice1
//        = weights[kernel_stride_z + 1 * kernel_one_output_slice + kernel_index
//        + slice];
//
//        float4 kernel_slice2 = weights[kernel_stride_z + 2 *
//        kernel_one_output_slice + kernel_index + slice];
//
//        float4 kernel_slice3 = weights[kernel_stride_z + 3 *
//        kernel_one_output_slice + kernel_index + slice];
//
//        input = inTexture.sample(sample, float2(input_x,    input_y), slice);
//        output.x += dot(input, kernel_slice);
//        output.y += dot(input, kernel_slice1);
//        output.z += dot(input, kernel_slice2);
//        output.w += dot(input, kernel_slice3);
//      }
//    }
//  }
//
//  outTexture.write(output, gid.xy, gid.z);
//}
//

kernel void conv_transpose3x3_stride2(
    texture2d_array<ftype, access::sample> inTexture[[texture(0)]],
    texture2d_array<ftype, access::sample> biasTexture[[texture(1)]],
    texture2d_array<ftype, access::write> outTexture[[texture(2)]],
    constant MetalConvTransposeParam& param[[buffer(0)]],
    const device ftype4* weights[[buffer(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    uint input_array_size = inTexture.get_array_size();
    uint weightTo = input_array_size * gid.z * 4 * 9;
    ftype4 w;
    ftype4 input1, input2, input3, input4;
    ftype4 output1, output2, output3, output4;
    uint ox = 2 * gid.x, oy = 2 * gid.y;
    if (param.hasAddOp == 1) {
        constant ElementwiseAddParam& addParam = param.addParam;
        output1 = get_bias(uint3(ox, oy, gid.z), addParam, biasTexture);
        output2 = get_bias(uint3(ox + 1, oy, gid.z), addParam, biasTexture);
        output3 = get_bias(uint3(ox, oy + 1, gid.z), addParam, biasTexture);
        output4 = get_bias(uint3(ox + 1, oy + 1, gid.z), addParam, biasTexture);
    } else {
        output1 = ftype4(0);
        output2 = ftype4(0);
        output3 = ftype4(0);
        output4 = ftype4(0);
    }

    for (ushort i = 0; i < input_array_size; ++i) {
        input1 = inTexture.sample(sample, float2(gid.x, gid.y), i);
        input2 = inTexture.sample(sample, float2(gid.x + 1, gid.y), i);
        input3 = inTexture.sample(sample, float2(gid.x, gid.y + 1), i);
        input4 = inTexture.sample(sample, float2(gid.x + 1, gid.y + 1), i);
        for (ushort j = 0; j < 4; ++j) {
            uint to = weightTo + j * 9 * input_array_size + i;
            w = weights[to + input_array_size * 0];
            output4[j] += dot(input4, w);
            w = weights[to + input_array_size * 1];
            output3[j] += dot(input3, w);
            w = weights[to + input_array_size * 2];
            output4[j] += dot(input3, w);
            w = weights[to + input_array_size * 3];
            output2[j] += dot(input2, w);
            w = weights[to + input_array_size * 4];
            output1[j] += dot(input1, w);
            w = weights[to + input_array_size * 5];
            output2[j] += dot(input1, w);
            w = weights[to + input_array_size * 6];
            output4[j] += dot(input2, w);
            w = weights[to + input_array_size * 7];
            output3[j] += dot(input1, w);
            w = weights[to + input_array_size * 8];
            output4[j] += dot(input1, w);
        }
    }
    outTexture.write(output1, uint2(ox, oy), gid.z);
    outTexture.write(output2, uint2(ox + 1, oy), gid.z);
    outTexture.write(output3, uint2(ox, oy + 1), gid.z);
    outTexture.write(output4, uint2(ox + 1, oy + 1), gid.z);
}

kernel void depthwise_conv_transpose3x3_stride2x2_half(
    texture2d_array<half, access::sample> inTexture[[texture(0)]],
    texture2d_array<half, access::sample> biasTexture[[texture(1)]],
    texture2d_array<half, access::write> outTexture[[texture(2)]],
    constant MetalConvTransposeParam& param[[buffer(0)]],
    const device half4* weights[[buffer(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    uint input_array_size = inTexture.get_array_size();
    half4 w;
    half4 input1, input2, input3, input4;
    half4 output1, output2, output3, output4;
    uint ox = 2 * gid.x, oy = 2 * gid.y;
    if (param.hasAddOp == 1) {
        constant ElementwiseAddParam& addParam = param.addParam;
        output1 = getBiasHalf(uint3(ox, oy, gid.z), addParam, biasTexture);
        output2 = getBiasHalf(uint3(ox + 1, oy, gid.z), addParam, biasTexture);
        output3 = getBiasHalf(uint3(ox, oy + 1, gid.z), addParam, biasTexture);
        output4 = getBiasHalf(uint3(ox + 1, oy + 1, gid.z), addParam, biasTexture);
    } else {
        output1 = half4(0);
        output2 = half4(0);
        output3 = half4(0);
        output4 = half4(0);
    }

    int i = gid.z;
    input1 = inTexture.sample(sample, float2(gid.x, gid.y), i);
    input2 = inTexture.sample(sample, float2(gid.x + 1, gid.y), i);
    input3 = inTexture.sample(sample, float2(gid.x, gid.y + 1), i);
    input4 = inTexture.sample(sample, float2(gid.x + 1, gid.y + 1), i);

    uint to = gid.z;
    w = weights[to + 0];
    output4 += input4 * w;
    to += input_array_size;
    w = weights[to];
    output3 += input3 * w;
    to += input_array_size;
    w = weights[to];
    output4 += input3 * w;
    to += input_array_size;
    w = weights[to];
    output2 += input2 * w;
    to += input_array_size;
    w = weights[to];
    output1 += input1 * w;
    to += input_array_size;
    w = weights[to];
    output2 += input1 * w;
    to += input_array_size;
    w = weights[to];
    output4 += input2 * w;
    to += input_array_size;
    w = weights[to];
    output3 += input1 * w;
    to += input_array_size;
    w = weights[to];
    output4 += input1 * w;
    outTexture.write(output1, uint2(ox, oy), gid.z);
    outTexture.write(output2, uint2(ox + 1, oy), gid.z);
    outTexture.write(output3, uint2(ox, oy + 1), gid.z);
    outTexture.write(output4, uint2(ox + 1, oy + 1), gid.z);
}

kernel void conv_transpose3x3_stride2x2_quadruple_half(
    texture2d_array<half, access::sample> inTexture[[texture(0)]],
    texture2d_array<half, access::sample> biasTexture[[texture(1)]],
    texture2d_array<half, access::write> outTexture[[texture(2)]],
    constant MetalConvTransposeParam& param[[buffer(0)]],
    const device half4* weights[[buffer(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    uint tx = gid.x << 1, ty = gid.y << 1;
    if (tx >= inTexture.get_width() || ty >= inTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    uint input_array_size = inTexture.get_array_size();
    uint weightTo = input_array_size * gid.z * 4 * 9;
    half4 w;
    half4 input1, input2, input3, input4, input5, input6, input7, input8, input9;
    half4 output1, output2, output3, output4, output5, output6, output7, output8, output9, output10,
        output11, output12, output13, output14, output15, output16;
    uint ox = 2 * tx, oy = 2 * ty;
    if (param.hasAddOp == 1) {
        constant ElementwiseAddParam& addParam = param.addParam;
        if (addParam.addByChannel == 1) {
            half4 b = getBiasHalf(uint3(ox, oy, gid.z), addParam, biasTexture);
            output1 = b;
            output2 = b;
            output3 = b;
            output4 = b;
            output5 = b;
            output6 = b;
            output7 = b;
            output8 = b;
            output9 = b;
            output10 = b;
            output11 = b;
            output12 = b;
            output13 = b;
            output14 = b;
            output15 = b;
            output16 = b;
        } else {
            output1 = getBiasHalf(uint3(ox, oy, gid.z), addParam, biasTexture);
            output2 = getBiasHalf(uint3(ox + 1, oy, gid.z), addParam, biasTexture);
            output3 = getBiasHalf(uint3(ox, oy + 1, gid.z), addParam, biasTexture);
            output4 = getBiasHalf(uint3(ox + 1, oy + 1, gid.z), addParam, biasTexture);

            output5 = getBiasHalf(uint3(ox + 2, oy, gid.z), addParam, biasTexture);
            output6 = getBiasHalf(uint3(ox + 3, oy, gid.z), addParam, biasTexture);
            output7 = getBiasHalf(uint3(ox + 2, oy + 1, gid.z), addParam, biasTexture);
            output8 = getBiasHalf(uint3(ox + 3, oy + 1, gid.z), addParam, biasTexture);

            output9 = getBiasHalf(uint3(ox, oy + 2, gid.z), addParam, biasTexture);
            output10 = getBiasHalf(uint3(ox + 1, oy + 2, gid.z), addParam, biasTexture);
            output11 = getBiasHalf(uint3(ox, oy + 3, gid.z), addParam, biasTexture);
            output12 = getBiasHalf(uint3(ox + 1, oy + 3, gid.z), addParam, biasTexture);

            output13 = getBiasHalf(uint3(ox + 2, oy + 2, gid.z), addParam, biasTexture);
            output14 = getBiasHalf(uint3(ox + 3, oy + 2, gid.z), addParam, biasTexture);
            output15 = getBiasHalf(uint3(ox + 2, oy + 3, gid.z), addParam, biasTexture);
            output16 = getBiasHalf(uint3(ox + 3, oy + 3, gid.z), addParam, biasTexture);
        }
    } else {
        output1 = half4(0);
        output2 = half4(0);
        output3 = half4(0);
        output4 = half4(0);
        output5 = half4(0);
        output6 = half4(0);
        output7 = half4(0);
        output8 = half4(0);
        output9 = half4(0);
        output10 = half4(0);
        output11 = half4(0);
        output12 = half4(0);
        output13 = half4(0);
        output14 = half4(0);
        output15 = half4(0);
        output16 = half4(0);
    }

    for (ushort i = 0; i < input_array_size; ++i) {
        input1 = inTexture.sample(sample, float2(gid.x, gid.y), i);
        input2 = inTexture.sample(sample, float2(gid.x + 1, gid.y), i);
        input3 = inTexture.sample(sample, float2(gid.x, gid.y + 1), i);
        input4 = inTexture.sample(sample, float2(gid.x + 1, gid.y + 1), i);
        for (ushort j = 0; j < 4; ++j) {
            uint to = weightTo + j * 9 * input_array_size + i;
            w = weights[to + input_array_size * 0];
            output4[j] += dot(input1, w);
            w = weights[to + input_array_size * 1];
            output3[j] += dot(input1, w);
            w = weights[to + input_array_size * 2];
            output4[j] += dot(input2, w);
            w = weights[to + input_array_size * 3];
            output2[j] += dot(input1, w);
            w = weights[to + input_array_size * 4];
            output1[j] += dot(input1, w);
            w = weights[to + input_array_size * 5];
            output2[j] += dot(input2, w);
            w = weights[to + input_array_size * 6];
            output4[j] += dot(input3, w);
            w = weights[to + input_array_size * 7];
            output3[j] += dot(input3, w);
            w = weights[to + input_array_size * 8];
            output4[j] += dot(input4, w);
        }
    }
    uint x = 2 * gid.x, y = 2 * gid.y;
    outTexture.write(output1, uint2(x, y), gid.z);
    outTexture.write(output2, uint2(x + 1, y), gid.z);
    outTexture.write(output3, uint2(x, y + 1), gid.z);
    outTexture.write(output4, uint2(x + 1, y + 1), gid.z);
}

kernel void conv_transpose3x3_caculate_half(
    texture2d_array<half, access::read> inTexture[[texture(0)]],
    texture2d_array<half, access::write> outTexture[[texture(1)]],
    const device half4* weights[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }

    uint input_array_size = inTexture.get_array_size();
    uint weightTo = input_array_size * gid.z * 4 * 9;
    float4 results[9] = {0};
    half4 w;
    for (ushort i = 0; i < input_array_size; ++i) {
        half4 input = inTexture.read(uint2(gid.x, gid.y), i);
        for (ushort j = 0; j < 9; ++j) {
            w = weights[weightTo + 0 * 9 * input_array_size + j * input_array_size + i];
            results[j].x += dot(float4(input), float4(w));
            w = weights[weightTo + 1 * 9 * input_array_size + j * input_array_size + i];
            results[j].y += dot(float4(input), float4(w));
            w = weights[weightTo + 2 * 9 * input_array_size + j * input_array_size + i];
            results[j].z += dot(float4(input), float4(w));
            w = weights[weightTo + 3 * 9 * input_array_size + j * input_array_size + i];
            results[j].w += dot(float4(input), float4(w));
        }
    }

    uint sx = gid.x * 3, sy = gid.y * 3;
    for (ushort h = 0; h < 3; ++h) {
        for (ushort w = 0; w < 3; ++w) {
            outTexture.write(half4(results[3 * h + w]), uint2(sx + w, sy + h), gid.z);
        }
    }
}

kernel void conv_transpose3x3_stride2_shift_left_half(
    texture2d_array<half, access::read> inTexture[[texture(0)]],
    texture2d_array<half, access::write> outTexture[[texture(1)]],
    ushort3 gid[[thread_position_in_grid]]) {
    ushort in_x = 3 * gid.x, out_x = 2 * gid.x, y = 3 * gid.y;
    if (out_x >= outTexture.get_width() || y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }
    // crop right
    //    for (ushort h = 0; h < 3; ++h) {
    //        ushort in_out_y = y + h;
    //        half4 add = in_x > 0 ? inTexture.read(uint2(in_x - 1, in_out_y),
    //        gid.z) : half4(0); half4 pix1 = inTexture.read(uint2(in_x,
    //        in_out_y), gid.z); half4 pix2 = inTexture.read(uint2(in_x + 1,
    //        in_out_y), gid.z); outTexture.write(pix1 + add, uint2(out_x,
    //        in_out_y), gid.z); outTexture.write(pix2, uint2(out_x + 1,
    //        in_out_y), gid.z);
    //    }

    // crop left
    for (ushort h = 0; h < 3; ++h) {
        ushort in_out_y = y + h;
        half4 add = (in_x + 3 >= inTexture.get_width())
                        ? half4(0)
                        : inTexture.read(uint2(in_x + 3, in_out_y), gid.z);
        half4 pix1 = inTexture.read(uint2(in_x + 1, in_out_y), gid.z);
        half4 pix2 = inTexture.read(uint2(in_x + 2, in_out_y), gid.z);
        outTexture.write(pix1, uint2(out_x, in_out_y), gid.z);
        outTexture.write(pix2 + add, uint2(out_x + 1, in_out_y), gid.z);
    }
}

kernel void conv_transpose3x3_stride2_shift_top_half(
    texture2d_array<half, access::read> inTexture[[texture(0)]],
    texture2d_array<half, access::write> outTexture[[texture(1)]],
    ushort3 gid[[thread_position_in_grid]]) {
    ushort in_y = 3 * gid.y, out_y = 2 * gid.y, x = 2 * gid.x;
    if (out_y >= outTexture.get_height() || x >= outTexture.get_width() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }

    // crop bottom
    //    for (ushort w = 0; w < 2; ++w) {
    //        ushort in_out_x = x + w;
    //        half4 add = in_y > 0 ? inTexture.read(uint2(in_out_x, in_y - 1),
    //        gid.z) : half4(0); half4 pix1 = inTexture.read(uint2(in_out_x,
    //        in_y), gid.z); half4 pix2 = inTexture.read(uint2(in_out_x, in_y +
    //        1), gid.z); outTexture.write(pix1 + add, uint2(in_out_x, out_y),
    //        gid.z); outTexture.write(pix2, uint2(in_out_x, out_y + 1), gid.z);
    //    }

    // crop top
    for (ushort w = 0; w < 2; ++w) {
        ushort in_out_x = x + w;
        half4 add = (in_y + 3 >= inTexture.get_height())
                        ? half4(0)
                        : inTexture.read(uint2(in_out_x, in_y + 3), gid.z);
        half4 pix1 = inTexture.read(uint2(in_out_x, in_y + 1), gid.z);
        half4 pix2 = inTexture.read(uint2(in_out_x, in_y + 2), gid.z);
        outTexture.write(pix1, uint2(in_out_x, out_y), gid.z);
        outTexture.write(pix2 + add, uint2(in_out_x, out_y + 1), gid.z);
    }
}

kernel void conv_transpose4x4_caculate_half(
    texture2d_array<half, access::read> inTexture[[texture(0)]],
    texture2d_array<half, access::write> outTexture[[texture(1)]],
    const device half4* weights[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }

    uint input_array_size = inTexture.get_array_size();
    uint weightTo = input_array_size * gid.z * 4 * 16;
    float4 results[16];
    for (ushort i = 0; i < 16; ++i) {
        results[i] = float4(0);
    }
    half4 w;
    for (ushort i = 0; i < input_array_size; ++i) {
        half4 input = inTexture.read(uint2(gid.x, gid.y), i);
        for (ushort j = 0; j < 16; ++j) {
            w = weights[weightTo + 0 * 16 * input_array_size + j * input_array_size + i];
            results[j].x += dot(float4(input), float4(w));
            w = weights[weightTo + 1 * 16 * input_array_size + j * input_array_size + i];
            results[j].y += dot(float4(input), float4(w));
            w = weights[weightTo + 2 * 16 * input_array_size + j * input_array_size + i];
            results[j].z += dot(float4(input), float4(w));
            w = weights[weightTo + 3 * 16 * input_array_size + j * input_array_size + i];
            results[j].w += dot(float4(input), float4(w));
        }
    }
    //    for (ushort j = 0; j < 16; ++j) {
    //        w = weights[input_array_size*4*4*128+j * input_array_size + gid.z];
    //        results[j] = float4(w);
    //    }

    ushort sx = gid.x * 4, sy = gid.y * 4;
    for (ushort h = 0; h < 4; ++h) {
        for (ushort w = 0; w < 4; ++w) {
            outTexture.write(half4(results[4 * h + w]), uint2(sx + w, sy + h), gid.z);
        }
    }
}

kernel void conv_transpose4x4_stride2_shift_left_half(
    texture2d_array<half, access::read> inTexture[[texture(0)]],
    texture2d_array<half, access::write> outTexture[[texture(1)]],
    ushort3 gid[[thread_position_in_grid]]) {
    ushort in_x = 4 * gid.x, out_x = 2 * gid.x, y = 4 * gid.y;
    if (out_x >= outTexture.get_width() || y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }
    // crop right
    //    for (ushort h = 0; h < 3; ++h) {
    //        ushort in_out_y = y + h;
    //        half4 add = in_x > 0 ? inTexture.read(uint2(in_x - 1, in_out_y),
    //        gid.z) : half4(0); half4 pix1 = inTexture.read(uint2(in_x,
    //        in_out_y), gid.z); half4 pix2 = inTexture.read(uint2(in_x + 1,
    //        in_out_y), gid.z); outTexture.write(pix1 + add, uint2(out_x,
    //        in_out_y), gid.z); outTexture.write(pix2, uint2(out_x + 1,
    //        in_out_y), gid.z);
    //    }

    // crop left
    for (ushort h = 0; h < 4; ++h) {
        ushort in_out_y = y + h;
        if (in_x == 0) {
            half4 pix1 = inTexture.read(uint2(1, in_out_y), gid.z);
            half4 pix2 = inTexture.read(uint2(inTexture.get_width() - 2, in_out_y), gid.z);
            outTexture.write(pix1, uint2(0, in_out_y), gid.z);
            outTexture.write(pix2, uint2(outTexture.get_width() - 1, in_out_y), gid.z);
        } else {
            half4 pix1 = inTexture.read(uint2(in_x, in_out_y), gid.z);
            half4 pix2 = inTexture.read(uint2(in_x + 1, in_out_y), gid.z);
            half4 pix3 = inTexture.read(uint2(in_x - 1, in_out_y), gid.z);
            half4 pix4 = inTexture.read(uint2(in_x - 2, in_out_y), gid.z);
            outTexture.write(pix1 + pix4, uint2(out_x - 1, in_out_y), gid.z);
            outTexture.write(pix2 + pix3, uint2(out_x, in_out_y), gid.z);
        }
    }
}

kernel void conv_transpose4x4_stride2_shift_top_half(
    texture2d_array<half, access::read> inTexture[[texture(0)]],
    texture2d_array<half, access::write> outTexture[[texture(1)]],
    ushort3 gid[[thread_position_in_grid]]) {
    ushort in_y = 4 * gid.y, out_y = 2 * gid.y, x = 2 * gid.x;
    if (out_y >= outTexture.get_height() || x >= outTexture.get_width() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }

    // crop bottom
    //    for (ushort w = 0; w < 2; ++w) {
    //        ushort in_out_x = x + w;
    //        half4 add = in_y > 0 ? inTexture.read(uint2(in_out_x, in_y - 1),
    //        gid.z) : half4(0); half4 pix1 = inTexture.read(uint2(in_out_x,
    //        in_y), gid.z); half4 pix2 = inTexture.read(uint2(in_out_x, in_y +
    //        1), gid.z); outTexture.write(pix1 + add, uint2(in_out_x, out_y),
    //        gid.z); outTexture.write(pix2, uint2(in_out_x, out_y + 1), gid.z);
    //    }

    // crop top
    for (ushort w = 0; w < 2; ++w) {
        ushort in_out_x = x + w;
        if (in_y == 0) {
            half4 pix1 = inTexture.read(uint2(in_out_x, 1), gid.z);
            half4 pix2 = inTexture.read(uint2(in_out_x, inTexture.get_height() - 2), gid.z);
            outTexture.write(pix1, uint2(in_out_x, 0), gid.z);
            outTexture.write(pix2, uint2(in_out_x, outTexture.get_height() - 1), gid.z);
        } else {
            half4 pix1 = inTexture.read(uint2(in_out_x, in_y), gid.z);
            half4 pix2 = inTexture.read(uint2(in_out_x, in_y + 1), gid.z);
            half4 pix3 = inTexture.read(uint2(in_out_x, in_y - 1), gid.z);
            half4 pix4 = inTexture.read(uint2(in_out_x, in_y - 2), gid.z);
            outTexture.write(pix1 + pix4, uint2(in_out_x, out_y - 1), gid.z);
            outTexture.write(pix2 + pix3, uint2(in_out_x, out_y), gid.z);
        }
    }
}
