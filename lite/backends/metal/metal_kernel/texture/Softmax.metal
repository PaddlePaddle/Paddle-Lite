/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

struct SoftmaxParam {
    int N;
    int K;
};

kernel void softmax(texture2d_array<ftype, access::read> inTexture [[texture(0)]],
                 texture2d_array<ftype, access::write> outTexture [[texture(1)]],
                 constant SoftmaxParam &sp [[buffer(0)]],
                 uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) {
        return;
    }
    int group = sp.K / 4;
    int remain = sp.K % 4;
#if LITE_WITH_METAL_FULL
    ftype max_value = FLT_MIN;
#else
    ftype max_value = HALF_MIN;
#endif
    
    // find max value
    for (int z = 0; z < group; z++) {
        ftype4 v = inTexture.read(uint2(gid.x, gid.y), z);
        max_value = max(v[0], max_value);
        max_value = max(v[1], max_value);
        max_value = max(v[2], max_value);
        max_value = max(v[3], max_value);
    }
    if (remain > 0) {
        ftype4 r = inTexture.read(uint2(gid.x, gid.y), group);
        for (int i = 0; i < remain; i++) {
            max_value = max(r[i], max_value);
        }
    }
    
    // need to consider the numerical error of overﬂow and underﬂow
    // doc: https://www.deeplearningbook.org/contents/numerical.html
    ftype4 sumv = {0, 0, 0, 0};
    for (int z = 0; z < group; z++) {
        ftype4 v = inTexture.read(uint2(gid.x, gid.y), z);
        sumv += exp(v - max_value);
    }
    ftype sum = sumv.x + sumv.y + sumv.z + sumv.w;
    if (remain > 0) {
        ftype4 r = inTexture.read(uint2(gid.x, gid.y), group);
        for (int i = 0; i < remain; i++) {
            sum += exp(r[i] - max_value);
        }
    }
    for (int z = 0; z <= group; z++) {
        ftype4 v = inTexture.read(uint2(gid.x, gid.y), z);
        v = exp(v - max_value);
        outTexture.write(v / sum, gid.xy, z);
    }
}
