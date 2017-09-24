/* Copyright (c) 2017 Baidu, Inc. All Rights Reserved.
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 ==============================================================================*/

#include <metal_stdlib>
using namespace metal;



kernel void mobilenet_preprocess(
                       texture2d<half, access::read> inTexture [[texture(0)]],
                       texture2d<half, access::write> outTexture [[texture(1)]],
                       uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height()) {
        return;
    }
    const auto means = float4(123.68f, 116.78f, 103.94f, 0.0f);
    const auto inColor = (float4(inTexture.read(gid)) * 255.0f - means) * 0.017f;
    outTexture.write(half4(inColor.z, inColor.y, inColor.x, 0.0f), gid);
    
}

kernel void squeezenet_preprocess(
                                 texture2d<half, access::read> inTexture [[texture(0)]],
                                 texture2d<half, access::write> outTexture [[texture(1)]],
                                 uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height()) {
        return;
    }
    const auto means = float4(104, 117, 123, 0.0);
    const auto inColor = (float4(inTexture.read(gid)) * 255.0f - means);
    outTexture.write(half4(inColor.z, inColor.y, inColor.x, 0.0f), gid);
    
}


