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

kernel void mul(texture2d_array<float, access::sample> inputX [[texture(0)]],
                            texture2d_array<float, access::sample> inputY [[texture(1)]],
                            texture2d_array<float, access::write> outTexture [[texture(2)]],
                            uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) return;
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    int xLen = inputX.get_width();
    float4 r = float4(0, 0, 0, 0);
    for (int i = 0; i < xLen; i++) {
        float4 iX = float4(inputX.sample(sample, float2(i, gid.y), 0));
        float4 iY1 = inputY.sample(sample, float2(gid.x, i*4), 0);
        float4 iY2 = inputY.sample(sample, float2(gid.x, i*4+1), 0);
        float4 iY3 = inputY.sample(sample, float2(gid.x, i*4+2), 0);
        float4 iY4 = inputY.sample(sample, float2(gid.x, i*4+3), 0);
        float4 tY1 = float4(iY1.x, iY2.x, iY3.x, iY4.x);
        float4 tY2 = float4(iY1.y, iY2.y, iY3.y, iY4.y);
        float4 tY3 = float4(iY1.z, iY2.z, iY3.z, iY4.z);
        float4 tY4 = float4(iY1.w, iY2.w, iY3.w, iY4.w);
        r.x += dot(iX, tY1);
        r.y += dot(iX, tY2);
        r.z += dot(iX, tY3);
        r.w += dot(iX, tY4);
    }
    outTexture.write(r, gid.xy, gid.z);
}

kernel void mul_half(texture2d_array<half, access::sample> inputX [[texture(0)]],
                                 texture2d_array<half, access::sample> inputY [[texture(1)]],
                                 texture2d_array<half, access::write> outTexture [[texture(2)]],
                                 uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) return;
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    int xLen = inputX.get_width();
    float4 r = float4(0, 0, 0, 0);
    for (int i = 0; i < xLen; i++) {
        float4 iX = float4(inputX.sample(sample, float2(i, gid.y), 0));
        half4 iY1 = inputY.sample(sample, float2(gid.x, i*4), 0);
        half4 iY2 = inputY.sample(sample, float2(gid.x, i*4+1), 0);
        half4 iY3 = inputY.sample(sample, float2(gid.x, i*4+2), 0);
        half4 iY4 = inputY.sample(sample, float2(gid.x, i*4+3), 0);
        float4 tY1 = float4(iY1.x, iY2.x, iY3.x, iY4.x);
        float4 tY2 = float4(iY1.y, iY2.y, iY3.y, iY4.y);
        float4 tY3 = float4(iY1.z, iY2.z, iY3.z, iY4.z);
        float4 tY4 = float4(iY1.w, iY2.w, iY3.w, iY4.w);
        r.x += dot(iX, tY1);
        r.y += dot(iX, tY2);
        r.z += dot(iX, tY3);
        r.w += dot(iX, tY4);
    }
    outTexture.write(half4(r), gid.xy, gid.z);
}

kernel void mul_add(texture2d_array<float, access::sample> inputX [[texture(0)]],
                            texture2d_array<float, access::sample> inputY [[texture(1)]],
                            texture2d_array<float, access::sample> biasY [[texture(2)]],
                            texture2d_array<float, access::write> outTexture [[texture(3)]],
                            uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) return;
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    int xLen = inputX.get_width();
    float4 r = float4(0, 0, 0, 0);
    for (int i = 0; i < xLen; i++) {
        float4 iX = float4(inputX.sample(sample, float2(i, gid.y), 0));
        float4 iY1 = inputY.sample(sample, float2(gid.x, i*4), 0);
        float4 iY2 = inputY.sample(sample, float2(gid.x, i*4+1), 0);
        float4 iY3 = inputY.sample(sample, float2(gid.x, i*4+2), 0);
        float4 iY4 = inputY.sample(sample, float2(gid.x, i*4+3), 0);
        float4 tY1 = float4(iY1.x, iY2.x, iY3.x, iY4.x);
        float4 tY2 = float4(iY1.y, iY2.y, iY3.y, iY4.y);
        float4 tY3 = float4(iY1.z, iY2.z, iY3.z, iY4.z);
        float4 tY4 = float4(iY1.w, iY2.w, iY3.w, iY4.w);
        r.x += dot(iX, tY1);
        r.y += dot(iX, tY2);
        r.z += dot(iX, tY3);
        r.w += dot(iX, tY4);
    }
    r += biasY.sample(sample, float2(gid.x, 0), 0);
    outTexture.write(r, gid.xy, gid.z);
}

kernel void mul_add_half(texture2d_array<half, access::sample> inputX [[texture(0)]],
                                 texture2d_array<half, access::sample> inputY [[texture(1)]],
                                 texture2d_array<half, access::sample> biasY [[texture(2)]],
                                 texture2d_array<half, access::write> outTexture [[texture(3)]],
                                 uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) return;
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    int xLen = inputX.get_width();
    float4 r = float4(0, 0, 0, 0);
    for (int i = 0; i < xLen; i++) {
        float4 iX = float4(inputX.sample(sample, float2(i, gid.y), 0));
        half4 iY1 = inputY.sample(sample, float2(gid.x, i*4), 0);
        half4 iY2 = inputY.sample(sample, float2(gid.x, i*4+1), 0);
        half4 iY3 = inputY.sample(sample, float2(gid.x, i*4+2), 0);
        half4 iY4 = inputY.sample(sample, float2(gid.x, i*4+3), 0);
        float4 tY1 = float4(iY1.x, iY2.x, iY3.x, iY4.x);
        float4 tY2 = float4(iY1.y, iY2.y, iY3.y, iY4.y);
        float4 tY3 = float4(iY1.z, iY2.z, iY3.z, iY4.z);
        float4 tY4 = float4(iY1.w, iY2.w, iY3.w, iY4.w);
        r.x += dot(iX, tY1);
        r.y += dot(iX, tY2);
        r.z += dot(iX, tY3);
        r.w += dot(iX, tY4);
    }
    r += float4(biasY.sample(sample, float2(gid.x, 0), 0));
    outTexture.write(half4(r), gid.xy, gid.z);
}
