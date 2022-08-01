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

#include "Common.metal"
using namespace metal;

struct ScaleParam {
    float scale;
    float abias;
    MetalActivationParam activationParam;
};

kernel void bias_after_scale(texture2d_array<ftype, access::read> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    constant ScaleParam& pm[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    const ftype4 input = inTexture.read(gid.xy, gid.z);
    const float scale = pm.scale;
    const float abias = pm.abias;
    const ftype4 output = scale * input + abias;
    ftype4 relu = activation(output, pm.activationParam);
    outTexture.write(relu, gid.xy, gid.z);
}

kernel void bias_before_scale(texture2d_array<ftype, access::read> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    constant ScaleParam& pm[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    const ftype4 input = inTexture.read(gid.xy, gid.z);
    const float scale = pm.scale;
    const float abias = pm.abias;
    const ftype4 output = scale * (input + abias);
    ftype4 relu = activation(output, pm.activationParam);
    outTexture.write(relu, gid.xy, gid.z);
}
