/* The following copyright is cited from Forge as an acknowledgement for its inspiring framework.
 Copyright (c) 2016-2017 M.I. Hollemans
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to
 deal in the Software without restriction, including without limitation the
 rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 IN THE SOFTWARE.
 */


#include <metal_stdlib>
using namespace metal;

enum NeuronType: ushort {
    NeuronTypeNone = 0,
    NeuronTypeReLU = 1,
    NeuronTypeLinear = 2,
    NeuronTypeSigmoid = 3,
    NeuronTypeTanH = 4,
    NeuronTypeAbsolute = 5,
    };
    
    constant ushort kernelWidth [[ function_constant(0) ]];
    constant ushort kernelHeight [[ function_constant(1) ]];
    constant ushort2 stride [[ function_constant(2) ]];
    constant ushort neuronType [[ function_constant(3) ]];
    
    struct KernelParams {
        ushort inputWidth;
        ushort inputHeight;
        ushort inputFeatureChannels;
        ushort inputSlices;
        ushort inputOffsetX;
        ushort inputOffsetY;
        ushort inputOffsetZ;
        ushort outputWidth;
        ushort outputHeight;
        ushort outputFeatureChannels;
        ushort outputSlices;
        ushort destinationSliceOffset;
        ushort outputOffsetX;
        ushort outputOffsetY;
        ushort outputOffsetZ;
        ushort edgeMode;
        float neuronA;
        float neuronB;
    };
    
    // Applying the activation function in the shader is quicker than creating
    // a new layer for it.
    inline float4 applyNeuron(float4 x, float a, float b) {
        if (neuronType == NeuronTypeReLU)
            return fmax(x, 0.0f) + a*fmin(x, 0.0f);
        if (neuronType == NeuronTypeLinear)
            return a*x + b;
        if (neuronType == NeuronTypeSigmoid)
            return 1.0f / (1.0f + exp(-x));
        if (neuronType == NeuronTypeTanH)
            return a * tanh(b * x);
        if (neuronType == NeuronTypeAbsolute)
            return fabs(x);
        return x;
    }
    
    inline half4 applyNeuron(half4 x, half a, half b) {
        if (neuronType == NeuronTypeReLU)
            return fmax(x, 0.0h) + a*fmin(x, 0.0h);
        if (neuronType == NeuronTypeLinear)
            return a*x + b;
        if (neuronType == NeuronTypeSigmoid)
            return 1.0h / (1.0h + exp(-x));
        if (neuronType == NeuronTypeTanH)
            return a * tanh(b * x);
        if (neuronType == NeuronTypeAbsolute)
            return fabs(x);
        return x;
    }

    
    kernel void depthwiseConv3x3(
                                 texture2d<half, access::sample> inTexture [[texture(0)]],
                                 texture2d<half, access::write> outTexture [[texture(1)]],
                                 constant KernelParams& params [[buffer(0)]],
                                 const device half* weights [[buffer(1)]],
                                 const device half4* biasTerms [[buffer(2)]],
                                 ushort2 gid [[thread_position_in_grid]])
    {
        if (gid.x >= outTexture.get_width() ||
            gid.y >= outTexture.get_height()) return;
        
        constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
        
        // Note: this is a very naive implementation of convolution.
        // There are ways to make it a lot faster...
        
        // Seen from the destination image, the stride is how far apart the pixels
        // are in the source image.
        const ushort2 pos = gid * stride + ushort2(params.inputOffsetX, params.inputOffsetY);
        
        // Read the 3x3 pixels surrounding the source pixel.
        // By processing the pixels as half4 values we do up to 4 channels at a time.
        half4 in[9];
        in[0] = inTexture.sample(s, float2(pos.x - 1, pos.y - 1));
        in[1] = inTexture.sample(s, float2(pos.x    , pos.y - 1));
        in[2] = inTexture.sample(s, float2(pos.x + 1, pos.y - 1));
        in[3] = inTexture.sample(s, float2(pos.x - 1, pos.y    ));
        in[4] = inTexture.sample(s, float2(pos.x    , pos.y    ));
        in[5] = inTexture.sample(s, float2(pos.x + 1, pos.y    ));
        in[6] = inTexture.sample(s, float2(pos.x - 1, pos.y + 1));
        in[7] = inTexture.sample(s, float2(pos.x    , pos.y + 1));
        in[8] = inTexture.sample(s, float2(pos.x + 1, pos.y + 1));
        
        const ushort kW = 3;
        const ushort kH = 3;
        
        // Multiply by the weights and put the weighted sum in the output pixel.
        // Do these calculations as 32-bit float or we lose too much precision.
        float4 out = float4(0.0f);
        for (ushort t = 0; t < kH*kW; ++t) {
            const auto pixel = float4(in[t]);
            out.x += pixel.x * float(weights[0*kH*kW + t]);
            out.y += pixel.y * float(weights[1*kH*kW + t]);
            out.z += pixel.z * float(weights[2*kH*kW + t]);
            out.w += pixel.w * float(weights[3*kH*kW + t]);
        }
        
        out += float4(biasTerms[0]);
        
        out = applyNeuron(out, params.neuronA, params.neuronB);
        
        outTexture.write(half4(out), gid);
    }
    
    kernel void depthwiseConv3x3_array(
                                       texture2d_array<half, access::sample> inTexture [[texture(0)]],
                                       texture2d_array<half, access::write> outTexture [[texture(1)]],
                                       constant KernelParams& params [[buffer(0)]],
                                       const device half* weights [[buffer(1)]],
                                       const device half4* biasTerms [[buffer(2)]],
                                       ushort3 gid [[thread_position_in_grid]])
    {
        if (gid.x >= outTexture.get_width() ||
            gid.y >= outTexture.get_height() ||
            gid.z >= outTexture.get_array_size()) return;
        
        constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
        
        const ushort2 pos = gid.xy * stride + ushort2(params.inputOffsetX, params.inputOffsetY);
        const ushort slice = gid.z;
        
        half4 in[9];
        in[0] = inTexture.sample(s, float2(pos.x - 1, pos.y - 1), slice);
        in[1] = inTexture.sample(s, float2(pos.x    , pos.y - 1), slice);
        in[2] = inTexture.sample(s, float2(pos.x + 1, pos.y - 1), slice);
        in[3] = inTexture.sample(s, float2(pos.x - 1, pos.y    ), slice);
        in[4] = inTexture.sample(s, float2(pos.x    , pos.y    ), slice);
        in[5] = inTexture.sample(s, float2(pos.x + 1, pos.y    ), slice);
        in[6] = inTexture.sample(s, float2(pos.x - 1, pos.y + 1), slice);
        in[7] = inTexture.sample(s, float2(pos.x    , pos.y + 1), slice);
        in[8] = inTexture.sample(s, float2(pos.x + 1, pos.y + 1), slice);
        
        const ushort kW = 3;
        const ushort kH = 3;
        
        float4 out = float4(0.0f);
        for (ushort t = 0; t < kH*kW; ++t) {
            const auto pixel = float4(in[t]);
            out.x += pixel.x * float(weights[(slice*4 + 0)*kH*kW + t]);
            out.y += pixel.y * float(weights[(slice*4 + 1)*kH*kW + t]);
            out.z += pixel.z * float(weights[(slice*4 + 2)*kH*kW + t]);
            out.w += pixel.w * float(weights[(slice*4 + 3)*kH*kW + t]);
        }
        
        out += float4(biasTerms[slice]);
        
        out = applyNeuron(out, params.neuronA, params.neuronB);
        
        outTexture.write(half4(out), gid.xy, gid.z);
    }

