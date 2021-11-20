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
using namespace metal;

kernel void prelu_channel(texture2d_array<float, access::sample> inTexture[[texture(0)]],
    texture2d_array<float, access::write> outTexture[[texture(1)]],
    const device float4* alpha[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }

    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    float4 input = inTexture.sample(sample, float2(gid.x, gid.y), gid.z);
    float4 alpha_value = alpha[gid.z];
    float4 output;
    output.x = input.x > 0 ? input.x : (alpha_value.x * input.x);
    output.y = input.y > 0 ? input.y : (alpha_value.y * input.y);
    output.z = input.z > 0 ? input.z : (alpha_value.z * input.z);
    output.w = input.w > 0 ? input.w : (alpha_value.w * input.w);
    outTexture.write(output, gid.xy, gid.z);
}

kernel void prelu_element(texture2d_array<float, access::sample> inTexture[[texture(0)]],
    texture2d_array<float, access::write> outTexture[[texture(1)]],
    const device float4* alpha[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }

    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    float4 input = inTexture.sample(sample, float2(gid.x, gid.y), gid.z);

    int alpha_to = (gid.y * inTexture.get_width() + gid.x) * inTexture.get_array_size();
    float4 alpha_value = alpha[alpha_to + gid.z];

    float4 output;
    output.x = input.x > 0 ? input.x : (alpha_value.x * input.x);
    output.y = input.y > 0 ? input.y : (alpha_value.y * input.y);
    output.z = input.z > 0 ? input.z : (alpha_value.z * input.z);
    output.w = input.w > 0 ? input.w : (alpha_value.w * input.w);
    outTexture.write(output, gid.xy, gid.z);
}

kernel void prelu_other(texture2d_array<float, access::sample> inTexture[[texture(0)]],
    texture2d_array<float, access::write> outTexture[[texture(1)]],
    const device float* alpha[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }

    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    float4 input = inTexture.sample(sample, float2(gid.x, gid.y), gid.z);
    float alpha_value = alpha[0];
    float4 output;
    output.x = input.x > 0 ? input.x : (alpha_value * input.x);
    output.y = input.y > 0 ? input.y : (alpha_value * input.y);
    output.z = input.z > 0 ? input.z : (alpha_value * input.z);
    output.w = input.w > 0 ? input.w : (alpha_value * input.w);
    outTexture.write(output, gid.xy, gid.z);
}

kernel void prelu_channel_half(texture2d_array<half, access::sample> inTexture[[texture(0)]],
    texture2d_array<half, access::write> outTexture[[texture(1)]],
    const device half4* alpha[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }

    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    half4 input = inTexture.sample(sample, float2(gid.x, gid.y), gid.z);
    half4 alpha_value = alpha[gid.z];
    half4 output;
    output.x = input.x > 0 ? input.x : (alpha_value.x * input.x);
    output.y = input.y > 0 ? input.y : (alpha_value.y * input.y);
    output.z = input.z > 0 ? input.z : (alpha_value.z * input.z);
    output.w = input.w > 0 ? input.w : (alpha_value.w * input.w);
    outTexture.write(output, gid.xy, gid.z);
}

kernel void prelu_element_half(texture2d_array<half, access::sample> inTexture[[texture(0)]],
    texture2d_array<half, access::write> outTexture[[texture(1)]],
    const device half4* alpha[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }

    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    half4 input = inTexture.sample(sample, float2(gid.x, gid.y), gid.z);

    int alpha_to = (gid.y * inTexture.get_width() + gid.x) * inTexture.get_array_size();
    half4 alpha_value = alpha[alpha_to + gid.z];

    half4 output;
    output.x = input.x > 0 ? input.x : (alpha_value.x * input.x);
    output.y = input.y > 0 ? input.y : (alpha_value.y * input.y);
    output.z = input.z > 0 ? input.z : (alpha_value.z * input.z);
    output.w = input.w > 0 ? input.w : (alpha_value.w * input.w);
    outTexture.write(output, gid.xy, gid.z);
}

kernel void prelu_other_half(texture2d_array<half, access::sample> inTexture[[texture(0)]],
    texture2d_array<half, access::write> outTexture[[texture(1)]],
    const device half* alpha[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }

    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    half4 input = inTexture.sample(sample, float2(gid.x, gid.y), gid.z);
    half alpha_value = alpha[0];
    half4 output;
    output.x = input.x > 0 ? input.x : (alpha_value * input.x);
    output.y = input.y > 0 ? input.y : (alpha_value * input.y);
    output.z = input.z > 0 ? input.z : (alpha_value * input.z);
    output.w = input.w > 0 ? input.w : (alpha_value * input.w);
    outTexture.write(output, gid.xy, gid.z);
}
