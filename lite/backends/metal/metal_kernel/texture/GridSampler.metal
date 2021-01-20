//
//  GridSamplerr.metal
//  paddle-mobile-metallib
//
//  Created by Li,Jian(MMS) on 2019/12/5.
//  Copyright © 2019 Ray. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

kernel void grid_sampler(texture2d_array<float, access::sample> inTexture [[texture(0)]],
                         texture2d_array<float, access::write> outTexture [[texture(1)]],
                         texture2d_array<float, access::read> gridTexture [[texture(2)]],
                         const device float2 *grid [[buffer(0)]],
                         uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    float4 g = gridTexture.read(gid.xy, 0);
    float x = (g.x + 1) * (inTexture.get_width() - 1) / 2;
    float y = (g.y + 1) * (inTexture.get_height() - 1) / 2;
    float x0 = floor(x);
    float y0 = floor(y);
    float4 input0 = inTexture.sample(sample, float2(x0,     y0),     gid.z);
    float4 input1 = inTexture.sample(sample, float2(x0 + 1, y0),     gid.z);
    float4 input2 = inTexture.sample(sample, float2(x0,     y0 + 1), gid.z);
    float4 input3 = inTexture.sample(sample, float2(x0 + 1, y0 + 1), gid.z);
    
    float4 output = input0 * (x0 + 1 - x) * (y0 + 1 - y) +
                    input1 * (x - x0) * (y0 + 1 - y) +
                    input2 * (x0 + 1 - x) * (y - y0) +
                    input3 * (x - x0) * (y - y0);
    
    outTexture.write(output, gid.xy, gid.z);
}

kernel void grid_sampler_half(texture2d_array<half, access::sample> inTexture [[texture(0)]],
                              texture2d_array<half, access::write> outTexture [[texture(1)]],
                              texture2d_array<float, access::read> gridTexture [[texture(2)]],
                              uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    float4 g = gridTexture.read(gid.xy, 0);
    float x = (g.x + 1) * (inTexture.get_width() - 1) / 2;
    float y = (g.y + 1) * (inTexture.get_height() - 1) / 2;
    float x0 = floor(x);
    float y0 = floor(y);
    float4 input0 = float4(inTexture.sample(sample, float2(x0,     y0),     gid.z));
    float4 input1 = float4(inTexture.sample(sample, float2(x0 + 1, y0),     gid.z));
    float4 input2 = float4(inTexture.sample(sample, float2(x0,     y0 + 1), gid.z));
    float4 input3 = float4(inTexture.sample(sample, float2(x0 + 1, y0 + 1), gid.z));
    
    float4 output = input0 * (x0 + 1 - x) * (y0 + 1 - y) +
                    input1 * (x - x0) * (y0 + 1 - y) +
                    input2 * (x0 + 1 - x) * (y - y0) +
                    input3 * (x - x0) * (y - y0);
    
    outTexture.write(half4(output), gid.xy, gid.z);
}
