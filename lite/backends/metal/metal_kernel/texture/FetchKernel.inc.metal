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

#ifdef P

#define CONCAT3_(a, b, c) a ## _ ## b ## _ ## c
#define CONCAT2_(a, b) a ## _ ## b
#define CONCAT2(a, b) a ## b
#define FUNC(m, n, q) CONCAT3_(m, n, q)
#define FUNC_T(m, n) CONCAT2_(m, n)

#define VECTOR(p, n) CONCAT2(p, n)

kernel void FUNC_T(fetch, P)(texture2d_array<P, access::read> inTexture [[texture(0)]],
                             device float *output [[buffer(0)]],
                             uint3 gid [[thread_position_in_grid]]) {
    uint input_width = inTexture.get_width();
    uint input_height = inTexture.get_height();
    if (gid.x >= input_width ||
        gid.y >= input_height ||
        gid.z >= inTexture.get_array_size()) {
        return;
    }
    
    const VECTOR(P, 4) input = inTexture.read(gid.xy, gid.z);
    uint delta = input_width * input_height;
    uint output_to = 4 * gid.z * delta + gid.y * input_width + gid.x;
    
    output[output_to] = input.x;
    output[output_to + delta] = input.y;
    output[output_to + 2 * delta] = input.z;
    output[output_to + 3 * delta] = input.w;
}

kernel void FUNC(fetch, 1or2, P)(texture2d_array<P, access::read> inTexture [[texture(0)]],
                                 device float4 *output [[buffer(0)]],
                                 uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= inTexture.get_width() ||
        gid.y >= inTexture.get_height() ||
        gid.z >= inTexture.get_array_size()) {
        return;
    }
    
    int input_width = inTexture.get_width();
    const VECTOR(P, 4) input = inTexture.read(gid.xy, gid.z);
    output[gid.y * input_width + gid.x] = float4(input);
}

#endif
