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
using namespace metal;

struct MetalConvParam {
    short offsetX;
    short offsetY;
    short offsetZ;
    ushort strideX;
    ushort strideY;
};


kernel void conv3x3(texture2d_array<half, access::sample> inTexture [[texture(0)]],
                    texture2d_array<half, access::write> outTexture [[texture(1)]],
                    constant MetalConvParam &param [[buffer(0)]],
                    const device half4 *weights [[buffer(1)]],
                    uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }
    
    short2 posInInput = short2(gid.xy) + short2(param.offsetX, param.offsetY);
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    const uint wightSliceCount = 36;
    uint weithTo = gid.z * wightSliceCount * inTexture.get_array_size();
    half4 output = 0.0;
    for (uint i = 0; i < inTexture.get_array_size(); ++i) {
        half4 input[9];
        input[0] = inTexture.sample(sample, float2(posInInput.x - 1, posInInput.y - 1), i);
        input[1] = inTexture.sample(sample, float2(posInInput.x, posInInput.y - 1), i);
        input[2] = inTexture.sample(sample, float2(posInInput.x + 1, posInInput.y - 1), i);
        input[3] = inTexture.sample(sample, float2(posInInput.x - 1, posInInput.y), i);
        input[4] = inTexture.sample(sample, float2(posInInput.x, posInInput.y), i);
        input[5] = inTexture.sample(sample, float2(posInInput.x + 1, posInInput.y), i);
        input[6] = inTexture.sample(sample, float2(posInInput.x - 1, posInInput.y + 1), i);
        input[7] = inTexture.sample(sample, float2(posInInput.x, posInInput.y + 1), i);
        input[8] = inTexture.sample(sample, float2(posInInput.x + 1, posInInput.y + 1), i);
        for (int j = 0; j < 9; ++j) {
            half4 weight = weights[weithTo + wightSliceCount * i + j * 4];
            output += dot(input[j], weight);
        }
    }
    outTexture.write(output, gid.xy, gid.z);
}

kernel void conv_add_batch_norm_relu_3x3(texture2d_array<half, access::sample> inTexture [[texture(0)]],
                                        texture2d_array<half, access::write> outTexture [[texture(1)]],
                                         constant MetalConvParam &param [[buffer(0)]],
                                         const device half4 *weights [[buffer(1)]],
                                         const device half4 *biase [[buffer(2)]],
                                         const device half4 *new_scale [[buffer(3)]],
                                         const device half4 *new_biase [[buffer(4)]],
                                         uint3 gid [[thread_position_in_grid]]) {
    
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }
    
    short2 posInInput = short2(gid.xy) + short2(param.offsetX, param.offsetY);
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    const uint wightSliceCount = 36;
    uint weithTo = gid.z * wightSliceCount * inTexture.get_array_size();
    half4 output = 0.0;
    for (uint i = 0; i < inTexture.get_array_size(); ++i) {
        half4 input[9];
        input[0] = inTexture.sample(sample, float2(posInInput.x - 1, posInInput.y - 1), i);
        input[1] = inTexture.sample(sample, float2(posInInput.x, posInInput.y - 1), i);
        input[2] = inTexture.sample(sample, float2(posInInput.x + 1, posInInput.y - 1), i);
        input[3] = inTexture.sample(sample, float2(posInInput.x - 1, posInInput.y), i);
        input[4] = inTexture.sample(sample, float2(posInInput.x, posInInput.y), i);
        input[5] = inTexture.sample(sample, float2(posInInput.x + 1, posInInput.y), i);
        input[6] = inTexture.sample(sample, float2(posInInput.x - 1, posInInput.y + 1), i);
        input[7] = inTexture.sample(sample, float2(posInInput.x, posInInput.y + 1), i);
        input[8] = inTexture.sample(sample, float2(posInInput.x + 1, posInInput.y + 1), i);
        for (int j = 0; j < 9; ++j) {
            half4 weight = weights[weithTo + wightSliceCount * i + j * 4];
            output += dot(input[j], weight);
        }
    }
    
    output = fmax((output + biase[gid.z]) * new_scale[gid.z] + new_biase[gid.z], 0.0h);
    outTexture.write(output, gid.xy, gid.z);
    
}



