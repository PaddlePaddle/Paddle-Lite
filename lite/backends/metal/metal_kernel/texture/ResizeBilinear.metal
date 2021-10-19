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
using namespace metal;

struct resize_bilinear_param {
    //  int32_t out_h;
    //  int32_t out_w;
    float ratio_h;
    float ratio_w;
};

kernel void resize_bilinear(texture2d_array<float, access::read> input[[texture(0)]],
    texture2d_array<float, access::write> output[[texture(2)]],
    constant resize_bilinear_param& pm[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    float4 r;
    if ((input.get_width() == output.get_width()) && (input.get_height() == output.get_height())) {
        r = input.read(gid.xy, gid.z);
    } else {
        float w = gid.x * pm.ratio_w;
        float h = gid.y * pm.ratio_h;
        uint w0 = w, h0 = h;
        uint w1 = w0 + 1, h1 = h0 + 1;
        float w1lambda = w - w0, h1lambda = h - h0;
        float w2lambda = 1.0 - w1lambda, h2lambda = 1.0 - h1lambda;
        if (w1 >= input.get_width()) w1 = w0;
        if (h1 >= input.get_height()) h1 = h0;
        float4 r0 = input.read(uint2(w0, h0), gid.z);
        float4 r1 = input.read(uint2(w1, h0), gid.z);
        float4 r2 = input.read(uint2(w0, h1), gid.z);
        float4 r3 = input.read(uint2(w1, h1), gid.z);
        r = h2lambda * (w2lambda * r0 + w1lambda * r1) + h1lambda * (w2lambda * r2 + w1lambda * r3);
    }
    output.write(r, gid.xy, gid.z);
}

kernel void resize_bilinear_half(texture2d_array<half, access::read> input[[texture(0)]],
    texture2d_array<half, access::write> output[[texture(2)]],
    constant resize_bilinear_param& pm[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    half4 r;
    if ((input.get_width() == output.get_width()) && (input.get_height() == output.get_height())) {
        r = input.read(gid.xy, gid.z);
    } else {
        half w = gid.x * pm.ratio_w;
        half h = gid.y * pm.ratio_h;
        uint w0 = w, h0 = h;
        uint w1 = w0 + 1, h1 = h0 + 1;
        half w1lambda = w - w0, h1lambda = h - h0;
        half w2lambda = 1.0 - w1lambda, h2lambda = 1.0 - h1lambda;
        if (w1 >= input.get_width()) w1 = w0;
        if (h1 >= input.get_height()) h1 = h0;
        half4 r0 = input.read(uint2(w0, h0), gid.z);
        half4 r1 = input.read(uint2(w1, h0), gid.z);
        half4 r2 = input.read(uint2(w0, h1), gid.z);
        half4 r3 = input.read(uint2(w1, h1), gid.z);
        r = h2lambda * (w2lambda * r0 + w1lambda * r1) + h1lambda * (w2lambda * r2 + w1lambda * r3);
    }
    output.write(r, gid.xy, gid.z);
    output.write(r, gid.xy, gid.z);
}
