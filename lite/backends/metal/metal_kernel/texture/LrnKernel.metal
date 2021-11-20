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

kernel void lrn_half(texture2d_array<half, access::sample> inTexture[[texture(0)]],
    texture2d_array<half, access::write> outTexture[[texture(1)]],
    constant LrnParam& param[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
    int start = max(0, (int)(gid.z - ((param.n - 1) / 2 + 3) / 4));
    int end = min(outTexture.get_array_size() - 1, gid.z + (param.n / 2 + 3) / 4);
    int startC = max(0, (int)(gid.z * 4 - (param.n - 1) / 2));
    int endC = min(param.channelN - 1, (int)(gid.z * 4 + param.n / 2));
    float4 res = float4(0);
    float4 input;
    for (int i = start; i <= end; i++) {
        float4 input = float4(inTexture.read(gid.xy, i));
        for (int j = 0; j < 4; j++) {
            int c = i * 4 + j;
            if (c >= startC && c <= endC) {
                res[0] += input[j] * input[j];
            }
            if (c >= startC + 1 && c <= min(param.channelN - 1, endC + 1)) {
                res[1] += input[j] * input[j];
            }
            if (c >= startC + 2 && c <= min(param.channelN - 1, endC + 2)) {
                res[2] += input[j] * input[j];
            }
            if (c >= startC + 3 && c <= min(param.channelN - 1, endC + 3)) {
                res[3] += input[j] * input[j];
            }
        }
    }
    input = float4(inTexture.read(gid.xy, gid.z));
    float4 output = input / pow(param.k + param.alpha * res, float4(param.beta));
    outTexture.write(half4(output), gid.xy, gid.z);
}

kernel void lrn(texture2d_array<float, access::sample> inTexture[[texture(0)]],
    texture2d_array<float, access::write> outTexture[[texture(1)]],
    constant LrnParam& param[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
    int start = max(0, (int)(gid.z - ((param.n - 1) / 2 + 3) / 4));
    int end = min(outTexture.get_array_size() - 1, gid.z + (param.n / 2 + 3) / 4);
    int startC = max(0, (int)(gid.z * 4 - (param.n - 1) / 2));
    int endC = min(param.channelN - 1, (int)(gid.z * 4 + param.n / 2));
    float4 res = float4(0);
    float4 input;
    for (int i = start; i <= end; i++) {
        float4 input = inTexture.read(gid.xy, i);
        for (int j = 0; j < 4; j++) {
            int c = i * 4 + j;
            if (c >= startC && c <= endC) {
                res[0] += input[j] * input[j];
            }
            if (c >= startC + 1 && c <= min(param.channelN - 1, endC + 1)) {
                res[1] += input[j] * input[j];
            }
            if (c >= startC + 2 && c <= min(param.channelN - 1, endC + 2)) {
                res[2] += input[j] * input[j];
            }
            if (c >= startC + 3 && c <= min(param.channelN - 1, endC + 3)) {
                res[3] += input[j] * input[j];
            }
        }
    }
    input = inTexture.read(gid.xy, gid.z);
    float4 output = input / pow(param.k + param.alpha * res, float4(param.beta));
    outTexture.write(output, gid.xy, gid.z);
}
