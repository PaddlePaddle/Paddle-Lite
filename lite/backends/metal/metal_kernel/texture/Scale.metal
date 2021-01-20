//
//  Scale.metal
//  paddle-mobile
//
//  Created by liuRuiLong on 2019/1/4.
//  Copyright © 2019 orange. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

kernel void scale(texture2d<float, access::sample> inTexture [[texture(0)]], texture2d<float, access::write> outTexture [[texture(1)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height()) return;
    float w_stride = inTexture.get_width() / outTexture.get_width();
    float h_stride = inTexture.get_height() / outTexture.get_height();
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    float4 input = inTexture.sample(sample, float2(gid.x * w_stride,    gid.y * h_stride), 0);
    outTexture.write(input, gid);
}

kernel void scale_half(texture2d<float, access::sample> inTexture [[texture(0)]], texture2d<half, access::write> outTexture [[texture(1)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height()) return;
    float w_stride = inTexture.get_width() / outTexture.get_width();
    float h_stride = inTexture.get_height() / outTexture.get_height();
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    float4 input = inTexture.sample(sample, float2(gid.x * w_stride,    gid.y * h_stride), 0);
    outTexture.write(half4(input), gid);
}
