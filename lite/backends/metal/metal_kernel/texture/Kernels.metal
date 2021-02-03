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

kernel void place_holder(texture2d<half, access::read> inTexture [[texture(0)]],
                         texture2d_array<half, access::write> outTexture [[texture(1)]],
                         uint3 gid [[thread_position_in_grid]]) {
}

struct OutputDim {
    ushort width;
    ushort height;
    ushort strideX;
    ushort strideY;
};

kernel void resize(texture2d<half, access::read> inTexture [[texture(0)]],
                   texture2d_array<half, access::write> outTexture [[texture(1)]],
                   constant OutputDim &params [[buffer(0)]],
                   uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) return;
    
    constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
    const uint2 pos = gid.xy * uint2(params.strideX, params.strideY);
    const half4 input = inTexture.read(pos);
    outTexture.write(half4(input.x, input.y, input.z, input.w), gid.xy, gid.z);
}


kernel void texture2d_to_2d_array(texture2d<float, access::read> inTexture [[texture(0)]],
                                  texture2d_array<float, access::write> outTexture [[texture(1)]],
                                  uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= inTexture.get_width() ||
        gid.y >= inTexture.get_height()){
        return;
    }
    const float4 input = inTexture.read(gid.xy);
    outTexture.write(input, gid.xy, 0);
}

kernel void texture2d_int_to_2d_array(texture2d<uint, access::read> inTexture [[texture(0)]],
                                      texture2d_array<float, access::write> outTexture [[texture(1)]],
                                      uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= inTexture.get_width() ||
        gid.y >= inTexture.get_height()){
        return;
    }
    const float4 input = (float4)inTexture.read(gid.xy);
    outTexture.write(input, gid.xy, 0);
}

kernel void texture2d_to_2d_array_half(texture2d<half, access::read> inTexture [[texture(0)]],
                                       texture2d_array<half, access::write> outTexture [[texture(1)]],
                                       uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= inTexture.get_width() ||
        gid.y >= inTexture.get_height()){
        return;
    }
    const half4 input = inTexture.read(gid.xy);
    outTexture.write(input, gid.xy, 0);
}

kernel void texture2d_int_to_2d_array_half(texture2d<uint, access::read> inTexture [[texture(0)]],
                                           texture2d_array<half, access::write> outTexture [[texture(1)]],
                                           uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= inTexture.get_width() ||
        gid.y >= inTexture.get_height()){
        return;
    }
    const half4 input = (half4)inTexture.read(gid.xy);
    outTexture.write(input, gid.xy, 0);
}
