//
//  Kernels.metal
//  paddle-mobile
//
//  Created by liuRuiLong on 2018/7/4.
//  Copyright © 2018年 orange. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

struct OutputDim {
    ushort width;
    ushort height;
    ushort strideX;
    ushort strideY;
};

kernel void resize(
                    texture2d<half, access::read> inTexture [[texture(0)]],
                    texture2d<half, access::write> outTexture [[texture(1)]],
                    constant OutputDim &params [[buffer(0)]],
                    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height()) {
        return;
    }
    
    constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
    const uint2 pos = gid.xy * uint2(params.strideX, params.strideY);
    const half4 input = inTexture.read(pos);
    outTexture.write(half4(input.x, input.y, input.z, 0.0h), gid);
}

