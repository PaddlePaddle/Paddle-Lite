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

kernel void exp(texture2d_array<float, access::sample> inTexture[[texture(0)]],
    texture2d_array<float, access::write> outTexture[[texture(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
    const float4 input = inTexture.read(gid.xy, gid.z);
    const float4 output = exp(input);
    outTexture.write(output, gid.xy, gid.z);
}

kernel void exp_half(texture2d_array<half, access::sample> inTexture[[texture(0)]],
    texture2d_array<half, access::write> outTexture[[texture(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
    const float4 input = float4(inTexture.read(gid.xy, gid.z));
    const float4 output = exp(input);
    outTexture.write(half4(output), gid.xy, gid.z);
}

kernel void relu(texture2d_array<ftype, access::read> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    const ftype4 input = inTexture.read(gid.xy, gid.z);
    const ftype4 relu = fmax(input, 0.0);
    outTexture.write(relu, gid.xy, gid.z);
}

kernel void relu6(texture2d_array<ftype, access::read> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    constant Relu6Param& pm[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    const ftype4 input = inTexture.read(gid.xy, gid.z);
    const float threshold = pm.threshold;
    const ftype4 relu = fmin(fmax(input, 0.0), threshold);
    outTexture.write(relu, gid.xy, gid.z);
}

kernel void leaky_relu(texture2d_array<ftype, access::read> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    constant LeakyReluParam& pm[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    const ftype4 input = inTexture.read(gid.xy, gid.z);
    const float alpha = pm.alpha;
    const ftype4 output = fmax(input, input * alpha);
    outTexture.write(output, gid.xy, gid.z);
}

kernel void sigmoid(texture2d_array<ftype, access::sample> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
    const ftype4 input = inTexture.read(gid.xy, gid.z);
    ftype4 output = 0.0;
    output.r = (input.r == 0.0) ? 0.0 : (1.0 / (1.0 + exp(-input.r)));
    output.g = (input.g == 0.0) ? 0.0 : (1.0 / (1.0 + exp(-input.g)));
    output.b = (input.b == 0.0) ? 0.0 : (1.0 / (1.0 + exp(-input.b)));
    output.a = (input.a == 0.0) ? 0.0 : (1.0 / (1.0 + exp(-input.a)));
    outTexture.write(output, gid.xy, gid.z);
}

kernel void hard_sigmoid(texture2d_array<ftype, access::sample> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    constant HardSigmoidParam& param[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
    const ftype4 input = inTexture.read(gid.xy, gid.z);
    const ftype4 output = clamp(input * param.slope + param.offset, 0.0, 1.0);
    outTexture.write(output, gid.xy, gid.z);
}

// activation function: swish
kernel void swish(texture2d_array<ftype, access::sample> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    constant SwishParam& param[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
    const ftype4 input = inTexture.read(gid.xy, gid.z);
    const ftype4 output = input / (1.0 + exp(-(input * param.beta)));
    outTexture.write(output, gid.xy, gid.z);
}

// activation function: hard swish
// output = input * (min(max(0, input + offset), threshold)) / scale
// please refer to
// https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/layers_cn/hard_swish_cn.html#hard-swish
kernel void hard_swish(texture2d_array<ftype, access::sample> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    constant HardSwishParam& param[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
    const ftype4 input = inTexture.read(gid.xy, gid.z);
    const ftype4 output =
        input * (fmin(fmax(0.0, input + param.offset), param.threshold)) / param.scale;
    outTexture.write(output, gid.xy, gid.z);
}

kernel void rsqrt(texture2d_array<float, access::sample> inTexture[[texture(0)]],
    texture2d_array<float, access::write> outTexture[[texture(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
    const float4 input = inTexture.read(gid.xy, gid.z);
    const float4 output = 1.0 / pow(input, 0.5);
    outTexture.write(output, gid.xy, gid.z);
}

kernel void rsqrt_half(texture2d_array<half, access::sample> inTexture[[texture(0)]],
    texture2d_array<half, access::write> outTexture[[texture(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
    const float4 input = float4(inTexture.read(gid.xy, gid.z));
    const float4 output = 1.0 / pow(input, 0.5);
    outTexture.write(half4(output), gid.xy, gid.z);
}

kernel void expand(texture2d_array<float, access::sample> inTexture[[texture(0)]],
    texture2d_array<float, access::write> outTexture[[texture(1)]],
    constant ExpandParam& pm[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
    if (pm.fast == 1) {
        const float4 input = inTexture.read(uint2(0, 0), gid.z);
        outTexture.write(input, gid.xy, gid.z);
    } else {
        uint c1 = (gid.z * 4) % pm.c;
        uint c2 = (gid.z * 4 + 1) % pm.c;
        uint c3 = (gid.z * 4 + 2) % pm.c;
        uint c4 = (gid.z * 4 + 3) % pm.c;
        uint w = gid.x % pm.w;
        uint h = gid.y % pm.h;
        const float4 input1 = inTexture.read(uint2(w, h), c1 / 4);
        const float4 input2 = inTexture.read(uint2(w, h), c2 / 4);
        const float4 input3 = inTexture.read(uint2(w, h), c3 / 4);
        const float4 input4 = inTexture.read(uint2(w, h), c4 / 4);
        outTexture.write(
            float4(input1[c1 % 4], input2[c2 % 4], input3[c3 % 4], input4[c4 % 4]), gid.xy, gid.z);
    }
}

kernel void expand_half(texture2d_array<half, access::sample> inTexture[[texture(0)]],
    texture2d_array<half, access::write> outTexture[[texture(1)]],
    constant ExpandParam& pm[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
    if (pm.fast == 1) {
        const half4 input = inTexture.read(uint2(0, 0), gid.z);
        outTexture.write(input, gid.xy, gid.z);
    } else {
        uint c1 = (gid.z * 4) % pm.c;
        uint c2 = (gid.z * 4 + 1) % pm.c;
        uint c3 = (gid.z * 4 + 2) % pm.c;
        uint c4 = (gid.z * 4 + 3) % pm.c;
        uint w = gid.x % pm.w;
        uint h = gid.y % pm.h;
        const half4 input1 = inTexture.read(uint2(w, h), c1 / 4);
        const half4 input2 = inTexture.read(uint2(w, h), c2 / 4);
        const half4 input3 = inTexture.read(uint2(w, h), c3 / 4);
        const half4 input4 = inTexture.read(uint2(w, h), c4 / 4);
        outTexture.write(
            half4(input1[c1 % 4], input2[c2 % 4], input3[c3 % 4], input4[c4 % 4]), gid.xy, gid.z);
    }
}
