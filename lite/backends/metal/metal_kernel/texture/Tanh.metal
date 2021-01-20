//
//  Tanh.metal
//  paddle-mobile-metallib
//
//  Created by Li,Jian(MMS) on 2019/8/8.
//  Copyright © 2019 Ray. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

kernel void tanh_half(texture2d_array<half, access::read> inTexture [[texture(0)]],
                      texture2d_array<half, access::write> outTexture [[texture(1)]],
                      uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) return;
    const half4 input = inTexture.read(gid.xy, gid.z);
    const float4 res = tanh((float4)input);
    outTexture.write(half4(res), gid.xy, gid.z);
}

kernel void tanh(texture2d_array<float, access::read> inTexture [[texture(0)]],
                 texture2d_array<float, access::write> outTexture [[texture(1)]],
                 uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) return;
    const float4 input = inTexture.read(gid.xy, gid.z);
    const float4 res = tanh(input);
    outTexture.write(float4(res), gid.xy, gid.z);
}


