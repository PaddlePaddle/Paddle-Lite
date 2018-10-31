//
//  BatchNormRelu.metal
//  paddle-mobile
//

#include <metal_stdlib>
using namespace metal;

struct MetalConvParam {
    short offsetX;
    short offsetY;
    short offsetZ;
    ushort strideX;
    ushort strideY;
};

kernel void batch_norm_relu_3x3(texture2d_array<float, access::sample> inTexture [[texture(0)]],
                                         texture2d_array<float, access::write> outTexture [[texture(1)]],
                                         const device float4 *new_scale [[buffer(0)]],
                                         const device float4 *new_biase [[buffer(1)]],
                                         uint3 gid [[thread_position_in_grid]]) {
    
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }
    
    float4 input;
    float4 output;
    constexpr sampler sample(coord::pixel, filter::nearest, address::clamp_to_zero);
    input = inTexture.sample(sample, gid.x, gid.y, gid.z);
    output = fmax(input * new_scale[gid.z] + new_biase[gid.z], 0.0);
    outTexture.write(output, gid.xy, gid.z);

}
