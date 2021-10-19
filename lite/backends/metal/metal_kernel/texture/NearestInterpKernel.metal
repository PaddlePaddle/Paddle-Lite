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

struct NearestInterpParam {
    float ratio_h;
    float ratio_w;
    float align_delta;
};

kernel void nearest_interp(texture2d_array<ftype, access::read> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    constant NearestInterpParam& param[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;
    float ratio_h = param.ratio_h;
    float ratio_w = param.ratio_w;
    //  if align center then align_delta=0.5, else align_delta=-1; calculate on CPU
    float align_delta = param.align_delta;

    uint x = uint(floor(gid.x * ratio_w + align_delta));
    uint y = uint(floor(gid.y * ratio_h + align_delta));
    ftype4 input = inTexture.read(uint2(x, y), gid.z);
    outTexture.write(input, gid.xy, gid.z);
}
