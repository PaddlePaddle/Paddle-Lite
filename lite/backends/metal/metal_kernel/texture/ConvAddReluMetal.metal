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

#pragma mark - conv

kernel void conv_1x1(texture2d_array<ftype, access::sample> inTexture[[texture(0)]],
    texture2d_array<ftype, access::sample> biasTexture[[texture(1)]],
    texture2d_array<ftype, access::write> outTexture[[texture(2)]],
    constant MetalConvParam& param[[buffer(0)]],
    const device ftype4* weights[[buffer(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }

    ushort2 stride = ushort2(param.strideX, param.strideY);
    ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);

    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    const uint kernelHXW = 1;

    uint input_arr_size = inTexture.get_array_size();
    uint weithTo = gid.z * kernelHXW * input_arr_size * 4;
    uint weights_len = kernelHXW * input_arr_size * param.oC - 1;

    ftype4 output = ftype4(0.0);
    if (param.hasAddOp) {
        output = get_bias(gid, param.addParam, biasTexture);
    }

    ftype4 input;
    for (uint i = 0; i < input_arr_size; ++i) {
        input = inTexture.sample(sample, float2(posInInput.x, posInInput.y), i);

        uint x = min(weithTo + 0 * kernelHXW * input_arr_size + i, weights_len);
        ftype4 weight_x = weights[x];
        output.x += dot(input, weight_x);

        uint y = min(weithTo + 1 * kernelHXW * input_arr_size + i, weights_len);
        ftype4 weight_y = weights[y];
        output.y += dot(input, weight_y);

        uint z = min(weithTo + 2 * kernelHXW * input_arr_size + i, weights_len);
        ftype4 weight_z = weights[z];
        output.z += dot(input, weight_z);

        uint w = min(weithTo + 3 * kernelHXW * input_arr_size + i, weights_len);
        ftype4 weight_w = weights[w];
        output.w += dot(input, weight_w);
    }

    ftype4 relu = activation(output, param.activationParam);
    outTexture.write(relu, gid.xy, gid.z);
}

kernel void conv_3x3(texture2d_array<ftype, access::sample> inTexture[[texture(0)]],
    texture2d_array<ftype, access::sample> biasTexture[[texture(1)]],
    texture2d_array<ftype, access::write> outTexture[[texture(2)]],
    constant MetalConvParam& param[[buffer(0)]],
    const device ftype4* weights[[buffer(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }

    ushort2 stride = ushort2(param.strideX, param.strideY);
    const ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);

    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    const uint kernelHXW = 9;
    uint input_arr_size = inTexture.get_array_size();
    uint weithTo = gid.z * kernelHXW * input_arr_size * 4;

    ftype4 output = ftype4(0.0);
    if (param.hasAddOp) {
        output = get_bias(gid, param.addParam, biasTexture);
    }

    ushort dilation_x = param.dilationX;
    ushort dilation_y = param.dilationY;

    ftype4 input[9];
    for (uint i = 0; i < input_arr_size; ++i) {
        input[0] = inTexture.sample(
            sample, float2(posInInput.x - dilation_x, posInInput.y - dilation_y), i);
        input[1] = inTexture.sample(sample, float2(posInInput.x, posInInput.y - dilation_y), i);
        input[2] = inTexture.sample(
            sample, float2(posInInput.x + dilation_x, posInInput.y - dilation_y), i);
        input[3] = inTexture.sample(sample, float2(posInInput.x - dilation_x, posInInput.y), i);
        input[4] = inTexture.sample(sample, float2(posInInput.x, posInInput.y), i);
        input[5] = inTexture.sample(sample, float2(posInInput.x + dilation_x, posInInput.y), i);
        input[6] = inTexture.sample(
            sample, float2(posInInput.x - dilation_x, posInInput.y + dilation_y), i);
        input[7] = inTexture.sample(sample, float2(posInInput.x, posInInput.y + dilation_y), i);
        input[8] = inTexture.sample(
            sample, float2(posInInput.x + dilation_x, posInInput.y + dilation_y), i);
        for (int j = 0; j < 9; ++j) {
            ftype4 weight_x =
                weights[weithTo + 0 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.x += dot(input[j], weight_x);

            ftype4 weight_y =
                weights[weithTo + 1 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.y += dot(input[j], weight_y);

            ftype4 weight_z =
                weights[weithTo + 2 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.z += dot(input[j], weight_z);

            ftype4 weight_w =
                weights[weithTo + 3 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.w += dot(input[j], weight_w);
        }
    }
    ftype4 relu = activation(output, param.activationParam);
    outTexture.write(relu, gid.xy, gid.z);
}

kernel void conv_7x7(texture2d_array<ftype, access::sample> inTexture[[texture(0)]],
    texture2d_array<ftype, access::sample> biasTexture[[texture(1)]],
    texture2d_array<ftype, access::write> outTexture[[texture(2)]],
    constant MetalConvParam& param[[buffer(0)]],
    const device ftype4* weights[[buffer(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }

    ushort2 stride = ushort2(param.strideX, param.strideY);
    const ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);

    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);

    const uint kernelHXW = 49;

    uint input_arr_size = inTexture.get_array_size();

    uint weithTo = gid.z * kernelHXW * input_arr_size * 4;

    ftype4 output = ftype4(0.0, 0.0, 0.0, 0.0);
    if (param.hasAddOp) {
        output = get_bias(gid, param.addParam, biasTexture);
    }

    ushort dilation_x = param.dilationX;
    ushort dilation_y = param.dilationY;

    ftype4 input[49];

    short x0 = posInInput.x - 3 * dilation_x;
    short x1 = posInInput.x - 2 * dilation_x;
    short x2 = posInInput.x - dilation_x;
    short x3 = posInInput.x;
    short x4 = posInInput.x + dilation_x;
    short x5 = posInInput.x + 2 * dilation_x;
    short x6 = posInInput.x + 3 * dilation_x;

    short y0 = posInInput.y - 3 * dilation_y;
    short y1 = posInInput.y - 2 * dilation_y;
    short y2 = posInInput.y - dilation_y;
    short y3 = posInInput.y;
    short y4 = posInInput.y + dilation_y;
    short y5 = posInInput.y + 2 * dilation_y;
    short y6 = posInInput.y + 3 * dilation_y;

    for (uint i = 0; i < input_arr_size; ++i) {
        input[0] = inTexture.sample(sample, float2(x0, y0), i);
        input[1] = inTexture.sample(sample, float2(x1, y0), i);
        input[2] = inTexture.sample(sample, float2(x2, y0), i);
        input[3] = inTexture.sample(sample, float2(x3, y0), i);
        input[4] = inTexture.sample(sample, float2(x4, y0), i);
        input[5] = inTexture.sample(sample, float2(x5, y0), i);
        input[6] = inTexture.sample(sample, float2(x6, y0), i);

        input[7] = inTexture.sample(sample, float2(x0, y1), i);
        input[8] = inTexture.sample(sample, float2(x1, y1), i);
        input[9] = inTexture.sample(sample, float2(x2, y1), i);
        input[10] = inTexture.sample(sample, float2(x3, y1), i);
        input[11] = inTexture.sample(sample, float2(x4, y1), i);
        input[12] = inTexture.sample(sample, float2(x5, y1), i);
        input[13] = inTexture.sample(sample, float2(x6, y1), i);

        input[14] = inTexture.sample(sample, float2(x0, y2), i);
        input[15] = inTexture.sample(sample, float2(x1, y2), i);
        input[16] = inTexture.sample(sample, float2(x2, y2), i);
        input[17] = inTexture.sample(sample, float2(x3, y2), i);
        input[18] = inTexture.sample(sample, float2(x4, y2), i);
        input[19] = inTexture.sample(sample, float2(x5, y2), i);
        input[20] = inTexture.sample(sample, float2(x6, y2), i);

        input[21] = inTexture.sample(sample, float2(x0, y3), i);
        input[22] = inTexture.sample(sample, float2(x1, y3), i);
        input[23] = inTexture.sample(sample, float2(x2, y3), i);
        input[24] = inTexture.sample(sample, float2(x3, y3), i);
        input[25] = inTexture.sample(sample, float2(x4, y3), i);
        input[26] = inTexture.sample(sample, float2(x5, y3), i);
        input[27] = inTexture.sample(sample, float2(x6, y3), i);

        input[28] = inTexture.sample(sample, float2(x0, y4), i);
        input[29] = inTexture.sample(sample, float2(x1, y4), i);
        input[30] = inTexture.sample(sample, float2(x2, y4), i);
        input[31] = inTexture.sample(sample, float2(x3, y4), i);
        input[32] = inTexture.sample(sample, float2(x4, y4), i);
        input[33] = inTexture.sample(sample, float2(x5, y4), i);
        input[34] = inTexture.sample(sample, float2(x6, y4), i);

        input[35] = inTexture.sample(sample, float2(x0, y5), i);
        input[36] = inTexture.sample(sample, float2(x1, y5), i);
        input[37] = inTexture.sample(sample, float2(x2, y5), i);
        input[38] = inTexture.sample(sample, float2(x3, y5), i);
        input[39] = inTexture.sample(sample, float2(x4, y5), i);
        input[40] = inTexture.sample(sample, float2(x5, y5), i);
        input[41] = inTexture.sample(sample, float2(x6, y5), i);

        input[42] = inTexture.sample(sample, float2(x0, y6), i);
        input[43] = inTexture.sample(sample, float2(x1, y6), i);
        input[44] = inTexture.sample(sample, float2(x2, y6), i);
        input[45] = inTexture.sample(sample, float2(x3, y6), i);
        input[46] = inTexture.sample(sample, float2(x4, y6), i);
        input[47] = inTexture.sample(sample, float2(x5, y6), i);
        input[48] = inTexture.sample(sample, float2(x6, y6), i);

        for (int j = 0; j < 49; ++j) {
            ftype4 weight_x =
                weights[weithTo + 0 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.x += dot(input[j], weight_x);

            ftype4 weight_y =
                weights[weithTo + 1 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.y += dot(input[j], weight_y);

            ftype4 weight_z =
                weights[weithTo + 2 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.z += dot(input[j], weight_z);

            ftype4 weight_w =
                weights[weithTo + 3 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.w += dot(input[j], weight_w);
        }
    }
    ftype4 relu = activation(output, param.activationParam);
    outTexture.write(relu, gid.xy, gid.z);
}

kernel void group_conv_3x3(texture2d_array<ftype, access::sample> inTexture[[texture(0)]],
    texture2d_array<ftype, access::sample> biasTexture[[texture(1)]],
    texture2d_array<ftype, access::write> outTexture[[texture(2)]],
    constant MetalConvParam& param[[buffer(0)]],
    const device ftype* weights[[buffer(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }

    ushort2 stride = ushort2(param.strideX, param.strideY);
    const ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);

    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);

    const uint kernelHXW = 9;

    ftype4 output = ftype4(0.0, 0.0, 0.0, 0.0);
    if (param.hasAddOp) {
        output = get_bias(gid, param.addParam, biasTexture);
    }

    ushort dilation_x = param.dilationX;
    ushort dilation_y = param.dilationY;

    ftype input[9];

    uint iC = param.iC, fC = param.fC, oC = param.oC;
    uint filter_array_size = (fC + 3) / 4;

    for (uint c = 0; c < 4; ++c) {
        uint output_depth = gid.z * 4 + c, output_c = output_depth % oC,
             output_n = output_depth / oC;
        for (uint i = 0; i < fC; ++i) {
            uint input_depth = output_n * iC + output_c * fC + i;
            uint input_array_index = input_depth / 4;
            uint input_array_item_index = input_depth % 4;
            input[0] = inTexture.sample(sample,
                float2(posInInput.x - dilation_x, posInInput.y - dilation_y),
                input_array_index)[input_array_item_index];
            input[1] = inTexture.sample(sample,
                float2(posInInput.x, posInInput.y - dilation_y),
                input_array_index)[input_array_item_index];
            input[2] = inTexture.sample(sample,
                float2(posInInput.x + dilation_x, posInInput.y - dilation_y),
                input_array_index)[input_array_item_index];
            input[3] = inTexture.sample(sample,
                float2(posInInput.x - dilation_x, posInInput.y),
                input_array_index)[input_array_item_index];
            input[4] = inTexture.sample(sample,
                float2(posInInput.x, posInInput.y),
                input_array_index)[input_array_item_index];
            input[5] = inTexture.sample(sample,
                float2(posInInput.x + dilation_x, posInInput.y),
                input_array_index)[input_array_item_index];
            input[6] = inTexture.sample(sample,
                float2(posInInput.x - dilation_x, posInInput.y + dilation_y),
                input_array_index)[input_array_item_index];
            input[7] = inTexture.sample(sample,
                float2(posInInput.x, posInInput.y + dilation_y),
                input_array_index)[input_array_item_index];
            input[8] = inTexture.sample(sample,
                float2(posInInput.x + dilation_x, posInInput.y + dilation_y),
                input_array_index)[input_array_item_index];
            for (int j = 0; j < 9; ++j) {
                ftype weight = weights[(output_c * kernelHXW + j) * filter_array_size * 4 + i];
                output[c] += input[j] * weight;
            }
        }
    }

    ftype4 relu = activation(output, param.activationParam);
    outTexture.write(relu, gid.xy, gid.z);
}

kernel void conv_2x2(texture2d_array<ftype, access::sample> inTexture[[texture(0)]],
    texture2d_array<ftype, access::sample> biasTexture[[texture(1)]],
    texture2d_array<ftype, access::write> outTexture[[texture(2)]],
    constant MetalConvParam& param[[buffer(0)]],
    const device ftype4* weights[[buffer(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }

    ushort2 stride = ushort2(param.strideX, param.strideY);
    const ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);

    const uint kernelHXW = 4;

    uint input_arr_size = inTexture.get_array_size();

    uint weithTo = gid.z * kernelHXW * input_arr_size * 4;

    ftype4 output = ftype4(0.0, 0.0, 0.0, 0.0);
    if (param.hasAddOp) {
        output = get_bias(gid, param.addParam, biasTexture);
    }

    ushort dilation_x = param.dilationX;
    ushort dilation_y = param.dilationY;

    ftype4 input[4];

    for (uint i = 0; i < input_arr_size; ++i) {
        input[0] = inTexture.sample(
            sample, float2(posInInput.x - dilation_x, posInInput.y - dilation_y), i);
        input[1] = inTexture.sample(sample, float2(posInInput.x, posInInput.y - dilation_y), i);
        input[2] = inTexture.sample(sample, float2(posInInput.x - dilation_x, posInInput.y), i);
        input[3] = inTexture.sample(sample, float2(posInInput.x, posInInput.y), i);

        for (int j = 0; j < 4; ++j) {
            ftype4 weight_x =
                weights[weithTo + 0 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.x += dot(input[j], weight_x);

            ftype4 weight_y =
                weights[weithTo + 1 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.y += dot(input[j], weight_y);

            ftype4 weight_z =
                weights[weithTo + 2 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.z += dot(input[j], weight_z);

            ftype4 weight_w =
                weights[weithTo + 3 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.w += dot(input[j], weight_w);
        }
    }
    ftype4 relu = activation(output, param.activationParam);
    outTexture.write(relu, gid.xy, gid.z);
}

kernel void conv_5x1(texture2d_array<ftype, access::sample> inTexture[[texture(0)]],
    texture2d_array<ftype, access::sample> biasTexture[[texture(1)]],
    texture2d_array<ftype, access::write> outTexture[[texture(2)]],
    constant MetalConvParam& param[[buffer(0)]],
    const device ftype4* weights[[buffer(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }

    ushort2 stride = ushort2(param.strideX, param.strideY);
    const ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);

    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);

    const uint kernelHXW = 5;

    uint input_arr_size = inTexture.get_array_size();

    uint weithTo = gid.z * kernelHXW * input_arr_size * 4;

    ftype4 output = ftype4(0.0, 0.0, 0.0, 0.0);
    if (param.hasAddOp) {
        output = get_bias(gid, param.addParam, biasTexture);
    }

    ushort dilation_y = param.dilationY;
    ftype4 input[5];

    for (uint i = 0; i < input_arr_size; ++i) {
        input[0] = inTexture.sample(sample, float2(posInInput.x, posInInput.y - 2 * dilation_y), i);

        input[1] = inTexture.sample(sample, float2(posInInput.x, posInInput.y - dilation_y), i);

        input[2] = inTexture.sample(sample, float2(posInInput.x, posInInput.y), i);

        input[3] = inTexture.sample(sample, float2(posInInput.x, posInInput.y + dilation_y), i);

        input[4] = inTexture.sample(sample, float2(posInInput.x, posInInput.y + 2 * dilation_y), i);

        for (int j = 0; j < 5; ++j) {
            ftype4 weight_x =
                weights[weithTo + 0 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.x += dot(input[j], weight_x);

            ftype4 weight_y =
                weights[weithTo + 1 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.y += dot(input[j], weight_y);

            ftype4 weight_z =
                weights[weithTo + 2 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.z += dot(input[j], weight_z);

            ftype4 weight_w =
                weights[weithTo + 3 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.w += dot(input[j], weight_w);
        }
    }
    ftype4 relu = activation(output, param.activationParam);
    outTexture.write(relu, gid.xy, gid.z);
}

kernel void conv_1x5(texture2d_array<ftype, access::sample> inTexture[[texture(0)]],
    texture2d_array<ftype, access::sample> biasTexture[[texture(1)]],
    texture2d_array<ftype, access::write> outTexture[[texture(2)]],
    constant MetalConvParam& param[[buffer(0)]],
    const device ftype4* weights[[buffer(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }

    ushort2 stride = ushort2(param.strideX, param.strideY);
    const ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);

    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);

    const uint kernelHXW = 5;

    uint input_arr_size = inTexture.get_array_size();

    uint weithTo = gid.z * kernelHXW * input_arr_size * 4;

    ftype4 output = ftype4(0.0, 0.0, 0.0, 0.0);
    if (param.hasAddOp) {
        output = get_bias(gid, param.addParam, biasTexture);
    }

    ushort dilation_x = param.dilationX;
    ftype4 input[5];

    for (uint i = 0; i < input_arr_size; ++i) {
        input[0] = inTexture.sample(sample, float2(posInInput.x - 2 * dilation_x, posInInput.y), i);

        input[1] = inTexture.sample(sample, float2(posInInput.x - dilation_x, posInInput.y), i);

        input[2] = inTexture.sample(sample, float2(posInInput.x, posInInput.y), i);

        input[3] = inTexture.sample(sample, float2(posInInput.x + dilation_x, posInInput.y), i);

        input[4] = inTexture.sample(sample, float2(posInInput.x + 2 * dilation_x, posInInput.y), i);

        for (int j = 0; j < 5; ++j) {
            ftype4 weight_x =
                weights[weithTo + 0 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.x += dot(input[j], weight_x);

            ftype4 weight_y =
                weights[weithTo + 1 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.y += dot(input[j], weight_y);

            ftype4 weight_z =
                weights[weithTo + 2 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.z += dot(input[j], weight_z);

            ftype4 weight_w =
                weights[weithTo + 3 * kernelHXW * input_arr_size + j * input_arr_size + i];
            output.w += dot(input[j], weight_w);
        }
    }
    ftype4 relu = activation(output, param.activationParam);
    outTexture.write(relu, gid.xy, gid.z);
}

#pragma mark - depthwise_conv

kernel void depthwise_conv_3x3(texture2d_array<ftype, access::sample> inTexture[[texture(0)]],
    texture2d_array<ftype, access::sample> biasTexture[[texture(1)]],
    texture2d_array<ftype, access::write> outTexture[[texture(2)]],
    constant MetalConvParam& param[[buffer(0)]],
    const device ftype* weights[[buffer(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }
    uint output_slice = gid.z;
    ushort2 stride = ushort2(param.strideX, param.strideY);
    ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    const uint kernelHXW = 9;
    uint weithTo = gid.z * kernelHXW * 4;

    ftype4 output = ftype4(0.0);
    if (param.hasAddOp) {
        output = get_bias(gid, param.addParam, biasTexture);
    }

    ushort dilation_x = param.dilationX;
    ushort dilation_y = param.dilationY;

    ftype4 inputs[9];
    inputs[0] = inTexture.sample(
        sample, float2(posInInput.x - dilation_x, posInInput.y - dilation_y), output_slice);
    inputs[1] =
        inTexture.sample(sample, float2(posInInput.x, posInInput.y - dilation_y), output_slice);
    inputs[2] = inTexture.sample(
        sample, float2(posInInput.x + dilation_x, posInInput.y - dilation_y), output_slice);
    inputs[3] =
        inTexture.sample(sample, float2(posInInput.x - dilation_x, posInInput.y), output_slice);
    inputs[4] = inTexture.sample(sample, float2(posInInput.x, posInInput.y), output_slice);
    inputs[5] =
        inTexture.sample(sample, float2(posInInput.x + dilation_x, posInInput.y), output_slice);
    inputs[6] = inTexture.sample(
        sample, float2(posInInput.x - dilation_x, posInInput.y + dilation_y), output_slice);
    inputs[7] =
        inTexture.sample(sample, float2(posInInput.x, posInInput.y + dilation_y), output_slice);
    inputs[8] = inTexture.sample(
        sample, float2(posInInput.x + dilation_x, posInInput.y + dilation_y), output_slice);
    for (int j = 0; j < 9; ++j) {
        ftype4 input = inputs[j];
        output.x += input.x * weights[weithTo + 0 * kernelHXW + j];
        output.y += input.y * weights[weithTo + 1 * kernelHXW + j];
        output.z += input.z * weights[weithTo + 2 * kernelHXW + j];
        output.w += input.w * weights[weithTo + 3 * kernelHXW + j];
    }

    ftype4 relu = activation(output, param.activationParam);
    outTexture.write(relu, gid.xy, gid.z);
}

kernel void depthwise_conv_3x3_unequal(
    texture2d_array<ftype, access::sample> inTexture[[texture(0)]],
    texture2d_array<ftype, access::sample> biasTexture[[texture(1)]],
    texture2d_array<ftype, access::write> outTexture[[texture(2)]],
    constant MetalConvParam& param[[buffer(0)]],
    const device ftype* weights[[buffer(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }
    uint inc = inTexture.get_array_size();
    uint inw = inTexture.get_width();
    uint inh = inTexture.get_height();
    uint output_slice = gid.z / 2;

    ushort2 stride = ushort2(param.strideX, param.strideY);
    ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    const uint kernelHXW = 9;
    uint weithTo = gid.z * kernelHXW * 4;

    ftype4 output;
    if (param.hasAddOp) {
        output = get_bias(gid, param.addParam, biasTexture);
    }

    ushort dilation_x = param.dilationX;
    ushort dilation_y = param.dilationY;
    float2 intput_sample[9];
    intput_sample[0] = float2(posInInput.x - dilation_x, posInInput.y - dilation_y);
    intput_sample[1] = float2(posInInput.x, posInInput.y - dilation_y);
    intput_sample[2] = float2(posInInput.x + dilation_x, posInInput.y - dilation_y);
    intput_sample[3] = float2(posInInput.x - dilation_x, posInInput.y);
    intput_sample[4] = float2(posInInput.x, posInInput.y);
    intput_sample[5] = float2(posInInput.x + dilation_x, posInInput.y);
    intput_sample[6] = float2(posInInput.x - dilation_x, posInInput.y + dilation_y);
    intput_sample[7] = float2(posInInput.x, posInInput.y + dilation_y);
    intput_sample[8] = float2(posInInput.x + dilation_x, posInInput.y + dilation_y);

    ftype4 input;
    for (int j = 0; j < 9; ++j) {
        // if(intput_sample[j].x >= 0 && intput_sample[j].x < inw && intput_sample[j].y >= 0 &&
        // intput_sample[j].y < inh){
        input = inTexture.sample(sample, intput_sample[j], output_slice);
        if (gid.z % 2 == 0) {
            output.x += input.x * weights[weithTo + 0 * kernelHXW + j];
            output.y += input.x * weights[weithTo + 1 * kernelHXW + j];
            output.z += input.y * weights[weithTo + 2 * kernelHXW + j];
            output.w += input.y * weights[weithTo + 3 * kernelHXW + j];
        } else {
            output.x += input.z * weights[weithTo + 0 * kernelHXW + j];
            output.y += input.z * weights[weithTo + 1 * kernelHXW + j];
            output.z += input.w * weights[weithTo + 2 * kernelHXW + j];
            output.w += input.w * weights[weithTo + 3 * kernelHXW + j];
        }
        //}
    }
    ftype4 relu = activation(output, param.activationParam);
    outTexture.write(relu, gid.xy, gid.z);
}

kernel void depthwise_conv_5x5(texture2d_array<ftype, access::sample> inTexture[[texture(0)]],
    texture2d_array<ftype, access::sample> biasTexture[[texture(1)]],
    texture2d_array<ftype, access::write> outTexture[[texture(2)]],
    constant MetalConvParam& param[[buffer(0)]],
    const device ftype* weights[[buffer(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }
    uint output_slice = gid.z;
    ushort2 stride = ushort2(param.strideX, param.strideY);
    ushort2 posInInput = ushort2(gid.xy) * stride + ushort2(param.offsetX, param.offsetY);
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    const uint kernelHXW = 25;
    uint weithTo = gid.z * kernelHXW * 4;

    ftype4 output = ftype4(0.0, 0.0, 0.0, 0.0);
    if (param.hasAddOp) {
        output = get_bias(gid, param.addParam, biasTexture);
    }

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            ftype4 input = inTexture.sample(
                sample, float2(posInInput.x + j - 2, posInInput.y + i - 2), output_slice);
            output.x += input.x * weights[weithTo + 0 * kernelHXW + 5 * i + j];
            output.y += input.y * weights[weithTo + 1 * kernelHXW + 5 * i + j];
            output.z += input.z * weights[weithTo + 2 * kernelHXW + 5 * i + j];
            output.w += input.w * weights[weithTo + 3 * kernelHXW + 5 * i + j];
        }
    }
    ftype4 relu = activation(output, param.activationParam);
    outTexture.write(relu, gid.xy, gid.z);
}

#pragma mark - winograd

kernel void conv_1x1_quadruple(texture2d_array<half, access::sample> inTexture[[texture(0)]],
    texture2d_array<half, access::sample> biasTexture[[texture(1)]],
    texture2d_array<half, access::write> outTexture[[texture(2)]],
    constant MetalConvParam& param[[buffer(0)]],
    const device half4* weights[[buffer(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    uint tx = gid.x << 1, ty = gid.y << 1, tz = gid.z;
    if (tx >= outTexture.get_width() || ty >= outTexture.get_height() ||
        tz >= outTexture.get_array_size()) {
        return;
    }

    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);

    int input_arr_size = inTexture.get_array_size();
    int weithTo = tz * input_arr_size * 4;

    half4 output0 = half4(0.0);
    half4 output1 = half4(0.0);
    half4 output2 = half4(0.0);
    half4 output3 = half4(0.0);

    if (param.hasAddOp) {
        output0 = get_bias(uint3(tx, ty, tz), param.addParam, biasTexture);
        output1 = get_bias(uint3(tx + 1, ty, tz), param.addParam, biasTexture);
        output2 = get_bias(uint3(tx, ty + 1, tz), param.addParam, biasTexture);
        output3 = get_bias(uint3(tx + 1, ty + 1, tz), param.addParam, biasTexture);
    }

    half4 input0, input1, input2, input3, f;
    for (int i = 0; i < input_arr_size; ++i) {
        input0 = inTexture.sample(sample, float2(tx, ty), i);
        input1 = inTexture.sample(sample, float2(tx + 1, ty), i);
        input2 = inTexture.sample(sample, float2(tx, ty + 1), i);
        input3 = inTexture.sample(sample, float2(tx + 1, ty + 1), i);

        f = weights[weithTo + i];
        output0.x += dot(input0, f);
        output1.x += dot(input1, f);
        output2.x += dot(input2, f);
        output3.x += dot(input3, f);

        f = weights[weithTo + input_arr_size + i];
        output0.y += dot(input0, f);
        output1.y += dot(input1, f);
        output2.y += dot(input2, f);
        output3.y += dot(input3, f);

        f = weights[weithTo + 2 * input_arr_size + i];
        output0.z += dot(input0, f);
        output1.z += dot(input1, f);
        output2.z += dot(input2, f);
        output3.z += dot(input3, f);

        f = weights[weithTo + 3 * input_arr_size + i];
        output0.w += dot(input0, f);
        output1.w += dot(input1, f);
        output2.w += dot(input2, f);
        output3.w += dot(input3, f);
    }

    half4 relu0 = activation(output0, param.activationParam);
    half4 relu1 = activation(output1, param.activationParam);
    half4 relu2 = activation(output2, param.activationParam);
    half4 relu3 = activation(output3, param.activationParam);

    outTexture.write(relu0, uint2(tx, ty), tz);
    outTexture.write(relu1, uint2(tx + 1, ty), tz);
    outTexture.write(relu2, uint2(tx, ty + 1), tz);
    outTexture.write(relu3, uint2(tx + 1, ty + 1), tz);
}

kernel void conv_3x3_winograd(texture2d_array<half, access::sample> inTexture[[texture(0)]],
    texture2d_array<half, access::sample> biasTexture[[texture(1)]],
    texture2d_array<half, access::write> outTexture[[texture(2)]],
    constant MetalConvParam& param[[buffer(0)]],
    const device half4x4* weights[[buffer(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    uint tx = gid.x << 1, ty = gid.y << 1, tc = gid.z;
    if (tx >= outTexture.get_width() || ty >= outTexture.get_height() ||
        tc >= outTexture.get_array_size()) {
        return;
    }
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    uint input_arr_size = inTexture.get_array_size();

    half4 output[4];
    output[0] = half4(0.0);
    output[1] = half4(0.0);
    output[2] = half4(0.0);
    output[3] = half4(0.0);

    if (param.hasAddOp == 1) {
        output[0] = get_bias(uint3(tx, ty, tc), param.addParam, biasTexture);
        output[1] = get_bias(uint3(tx + 1, ty, tc), param.addParam, biasTexture);
        output[2] = get_bias(uint3(tx, ty + 1, tc), param.addParam, biasTexture);
        output[3] = get_bias(uint3(tx + 1, ty + 1, tc), param.addParam, biasTexture);
    }

    half4 input[16];
    for (uint i = 0; i < input_arr_size; ++i) {
        input[0] = inTexture.sample(sample, float2(tx - 1, ty - 1), i);
        input[1] = inTexture.sample(sample, float2(tx, ty - 1), i);
        input[2] = inTexture.sample(sample, float2(tx + 1, ty - 1), i);
        input[3] = inTexture.sample(sample, float2(tx + 2, ty - 1), i);

        input[4] = inTexture.sample(sample, float2(tx - 1, ty), i);
        input[5] = inTexture.sample(sample, float2(tx, ty), i);
        input[6] = inTexture.sample(sample, float2(tx + 1, ty), i);
        input[7] = inTexture.sample(sample, float2(tx + 2, ty), i);

        input[8] = inTexture.sample(sample, float2(tx - 1, ty + 1), i);
        input[9] = inTexture.sample(sample, float2(tx, ty + 1), i);
        input[10] = inTexture.sample(sample, float2(tx + 1, ty + 1), i);
        input[11] = inTexture.sample(sample, float2(tx + 2, ty + 1), i);

        input[12] = inTexture.sample(sample, float2(tx - 1, ty + 2), i);
        input[13] = inTexture.sample(sample, float2(tx, ty + 2), i);
        input[14] = inTexture.sample(sample, float2(tx + 1, ty + 2), i);
        input[15] = inTexture.sample(sample, float2(tx + 2, ty + 2), i);

        for (uint otc = 0; otc < 4; otc++) {
            uint weightTo = (tc * 4 + otc) * param.iC;
            if (weightTo >= param.oC * param.iC) {
                break;
            }
            for (int c = 0; c < 4; ++c) {
                if (i * 4 + c >= param.iC) {
                    break;
                }
                uint toC = weightTo + i * 4 + c;
                if (toC >= param.oC * param.iC) {
                    break;
                }
                half I[16];
                for (uint j = 0; j < 16; ++j) {
                    I[j] = input[j][c];
                }
                half4x4 f = weights[toC];
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
                output[0][otc] += T[0] + T[4] + T[8] + tmp1 + tmp2;
                output[1][otc] += T[3] + T[7] + T[11] + tmp1 - tmp2;
                tmp1 = T[5] - T[9] + T[13];
                tmp2 = T[6] - T[10] + T[14];
                output[2][otc] += T[4] - T[8] + T[12] + tmp1 + tmp2;
                output[3][otc] += T[7] - T[11] + T[15] + tmp1 - tmp2;
            }
        }
    }

    half4 relu0 = activation(output[0], param.activationParam);
    half4 relu1 = activation(output[1], param.activationParam);
    half4 relu2 = activation(output[2], param.activationParam);
    half4 relu3 = activation(output[3], param.activationParam);

    outTexture.write(relu0, uint2(tx, ty), tc);
    outTexture.write(relu1, uint2(tx + 1, ty), tc);
    outTexture.write(relu2, uint2(tx, ty + 1), tc);
    outTexture.write(relu3, uint2(tx + 1, ty + 1), tc);
}

kernel void depthwise_conv_3x3_winograd(
    texture2d_array<half, access::sample> inTexture[[texture(0)]],
    texture2d_array<half, access::sample> biasTexture[[texture(1)]],
    texture2d_array<half, access::write> outTexture[[texture(2)]],
    constant MetalConvParam& param[[buffer(0)]],
    const device half4x4* weights[[buffer(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    uint x = gid.x, y = gid.y, z = gid.z;

    uint tx = x << 1;
    uint ty = y << 1;
    uint tc = z;

    if (tx >= outTexture.get_width() || ty >= outTexture.get_height() ||
        tc >= outTexture.get_array_size()) {
        return;
    }

    int hasComputedC = 4 * tc;

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

    int weightTo = 4 * tc;
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
        half4 base = get_bias(uint3(tx, ty, tc), param.addParam, biasTexture);
        res[0] += base;
        base = get_bias(uint3(tx + 1, ty, tc), param.addParam, biasTexture);
        res[1] += base;
        base = get_bias(uint3(tx, ty + 1, tc), param.addParam, biasTexture);
        res[2] += base;
        base = get_bias(uint3(tx + 1, ty + 1, tc), param.addParam, biasTexture);
        res[3] += base;
    }

    half4 relu0 = activation(res[0], param.activationParam);
    half4 relu1 = activation(res[1], param.activationParam);
    half4 relu2 = activation(res[2], param.activationParam);
    half4 relu3 = activation(res[3], param.activationParam);

    outTexture.write(relu0, uint2(tx, ty), tc);
    outTexture.write(relu1, uint2(tx + 1, ty), tc);
    outTexture.write(relu2, uint2(tx, ty + 1), tc);
    outTexture.write(relu3, uint2(tx + 1, ty + 1), tc);
}

#pragma mark -
