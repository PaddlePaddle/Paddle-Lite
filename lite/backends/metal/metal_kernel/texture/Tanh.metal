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

kernel void tanh_half(texture2d_array<half, access::read> inTexture[[texture(0)]],
    texture2d_array<half, access::write> outTexture[[texture(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    const half4 input = inTexture.read(gid.xy, gid.z);
    const float4 res = tanh((float4)input);
    outTexture.write(half4(res), gid.xy, gid.z);
}

kernel void tanh(texture2d_array<float, access::read> inTexture[[texture(0)]],
    texture2d_array<float, access::write> outTexture[[texture(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    const float4 input = inTexture.read(gid.xy, gid.z);
    const float4 res = tanh(input);
    outTexture.write(float4(res), gid.xy, gid.z);
}
