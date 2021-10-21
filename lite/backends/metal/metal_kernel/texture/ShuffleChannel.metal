#include <metal_stdlib>

#include "Common.metal"
using namespace metal;

kernel void shuffle_channel(texture2d_array<ftype, access::read> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    constant ShuffleChannelParam& param[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
    const ftype4 input = inTexture.read(gid.xy, gid.z);

    const uint group = param.group;
    const uint group_per_channel = param.channel_per_group;

    ftype4 output;
    for (int i = 0; i < 4; ++i) {
        uint out_ch_idx = gid.z * 4 + i;
        uint in_ch_idx = out_ch_idx % group * group_per_channel + out_ch_idx / group;
        uint input_z = in_ch_idx >> 2;
        const ftype4 input = inTexture.read(gid.xy, input_z);
        output[i] = input[in_ch_idx % 4];
    }
    outTexture.write(output, gid.xy, gid.z);
}