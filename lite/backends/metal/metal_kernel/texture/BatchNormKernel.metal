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

#include "Common.metal"
#include <metal_stdlib>

using namespace metal;

kernel void batchnorm(texture2d_array<ftype, access::read> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    const device ftype4* nscale[[buffer(0)]],
    const device ftype4* nbias[[buffer(1)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;

    const ftype4 input = inTexture.read(gid.xy, gid.z);
    ftype4 output = input * nscale[gid.z] + nbias[gid.z];
    outTexture.write(output, gid.xy, gid.z);
}
