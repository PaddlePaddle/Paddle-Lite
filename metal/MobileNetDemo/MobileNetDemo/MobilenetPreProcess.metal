//
//  MobilenetProcess.metal
//  MobileNetDemo
//
//  Created by liuRuiLong on 2019/1/5.
//  Copyright © 2019 Ray. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

kernel void mobilenet_preprocess(
                                 texture2d<float, access::read> inTexture [[texture(0)]],
                                 texture2d<float, access::write> outTexture [[texture(1)]],
                                 uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height()) {
        return;
    }
    const auto means = float4(123.68f, 116.78f, 103.94f, 0.0f);
    const float4 inColor = (inTexture.read(gid) * 255.0 - means) * 0.017;
    outTexture.write(float4(inColor.z, inColor.y, inColor.x, 0.0f), gid);
}

kernel void mobilenet_preprocess_half(
                                      texture2d<half, access::read> inTexture [[texture(0)]],
                                      texture2d<half, access::write> outTexture [[texture(1)]],
                                      uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height()) {
        return;
    }
    const auto means = half4(123.68f, 116.78f, 103.94f, 0.0f);
    const half4 inColor = (inTexture.read(gid) * 255.0 - means) * 0.017;
    outTexture.write(half4(inColor.z, inColor.y, inColor.x, 0.0f), gid);
}
