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

kernel void mobilenet_ssd_preprocess(
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

kernel void mobilenet_ssd_preprocess_half(
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

kernel void genet_preprocess(texture2d<float, access::read> inTexture [[texture(0)]], texture2d<float, access::write> outTexture [[texture(1)]], uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height()) {
        return;
    }
    const auto means = float4(128.0f, 128.0f, 128.0f, 0.0f);
    const float4 inColor = (inTexture.read(gid) * 255.0 - means) * 0.017;
    outTexture.write(float4(inColor.z, inColor.y, inColor.x, 0.0f), gid);
}

kernel void genet_preprocess_half(texture2d<half, access::read> inTexture [[texture(0)]], texture2d<half, access::write> outTexture [[texture(1)]], uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height()) {
        return;
    }
    const auto means = half4(128.0f, 128.0f, 128.0f, 0.0f);
    const half4 inColor = (inTexture.read(gid) * 255.0 - means) * 0.017;
    outTexture.write(half4(inColor.z, inColor.y, inColor.x, 0.0f), gid);
}

kernel void mobilent_ar_preprocess(texture2d<float, access::read> inTexture [[texture(0)]], texture2d<float, access::write> outTexture [[texture(1)]], uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height()) {
        return;
    }
    const auto means = float4(128.0f, 128.0f, 128.0f, 0.0f);
    const float4 inColor = (inTexture.read(gid) * 255.0 - means) * 0.017;
    outTexture.write(float4(inColor.z, inColor.y, inColor.x, 0.0f), gid);
}

kernel void mobilent_ar_preprocess_half(texture2d<half, access::read> inTexture [[texture(0)]], texture2d<half, access::write> outTexture [[texture(1)]], uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height()) {
        return;
    }
    const auto means = half4(128.0f, 128.0f, 128.0f, 0.0f);
    const half4 inColor = (inTexture.read(gid) * 255.0 - means) * 0.017;
    outTexture.write(half4(inColor.z, inColor.y, inColor.x, 0.0f), gid);
}
