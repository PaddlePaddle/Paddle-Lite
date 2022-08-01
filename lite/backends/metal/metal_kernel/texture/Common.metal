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

#pragma mark -

#if LITE_WITH_METAL_FULL
typedef float ftype;
typedef float2 ftype2;
typedef float3 ftype3;
typedef float4 ftype4;
typedef float2x2 ftype2x2;
typedef float2x3 ftype2x3;
typedef float2x4 ftype2x4;
typedef float3x2 ftype3x2;
typedef float3x3 ftype3x3;
typedef float3x4 ftype3x4;
typedef float4x2 ftype4x2;
typedef float4x3 ftype4x3;
typedef float4x4 ftype4x4;
#else
typedef half ftype;
typedef half2 ftype2;
typedef half3 ftype3;
typedef half4 ftype4;
typedef half2x2 ftype2x2;
typedef half2x3 ftype2x3;
typedef half2x4 ftype2x4;
typedef half3x2 ftype3x2;
typedef half3x3 ftype3x3;
typedef half3x4 ftype3x4;
typedef half4x2 ftype4x2;
typedef half4x3 ftype4x3;
typedef half4x4 ftype4x4;
#endif

#pragma mark -

enum ActivationType : ushort {
    NONE = 0,
    RELU = 1,
    RELU6 = 2,
    PRELU = 3,
    LEAKY_RELU = 4,
    HARD_SIGMOID = 5,
    HARD_SWISH = 10,
};

struct DropoutParam {
    float scale;
};

struct MetalActivationParam {
    ActivationType activationType;
    float threshold;  // RELU6
    float alpha;      // LEAKY_RELU
    float offset;     // HARD_SIGMOID
    float slope;      // HARD_SIGMOID
    float scale;      // HARD_SWISH
};

struct ElementwiseAddParam {
    int32_t fast;          // wise element add
    int32_t addByChannel;  // only C channell
    int32_t axis;          // input_y index at input_x
    int32_t ylen;          // input_y axis
    int32_t xdim[4];       // input_x dim [NCHW-CPU]-> [NHWC-GPU]
    int32_t xtrans[4];     // input_x transpose dim
    int32_t ydim[4];       // input_y dim on gpu
    int32_t ytrans[4];     // input_x transpose dim
    int32_t ByNum;         // only one number
    int32_t ByHW;          // only HW
    int32_t ByW;           // only W
    int32_t arithmetic_type;
};

struct MatmulParam {
    bool xtrans;
    bool ytrans;
    bool broadcast;
};

struct ElementwiseParam {
    int32_t byChannel;
};

struct MetalConvParam {
    short offsetX;
    short offsetY;
    short offsetZ;
    ushort strideX;
    ushort strideY;
    ushort dilationX;
    ushort dilationY;
    ushort groups;
    ushort iC;
    ushort fC;
    ushort oC;
    ushort hasAddOp;
    ushort hasReluOp;
    ElementwiseAddParam addParam;
    MetalActivationParam activationParam;
};

struct Pad2dParam {
    short paddingTop;
    short paddingBottom;
    short paddingLeft;
    short paddingRight;
    float padValue;
    short mode;
};

struct MetalInstanceNormReluParam {
    ushort hasReluOp;
};

struct MetalConvTransposeParam {
    ushort kernelW;
    ushort kernelH;

    ushort strideX;
    ushort strideY;

    ushort paddingX;
    ushort paddingY;

    ushort dilationX;
    ushort dilationY;

    ushort groups;
    ushort iC;
    ushort fC;
    ushort oC;

    ushort hasAddOp;
    ElementwiseAddParam addParam;
    MetalActivationParam activationParam;
};

struct LrnParam {
    int32_t n;
    int32_t channelN;
    float k;
    float alpha;
    float beta;
};

struct PixelShuffleParam {
    int32_t upscale_factor;
};

struct ExpandParam {
    ushort fast;
    ushort c;
    ushort h;
    ushort w;
};

struct B2TParam {
    int32_t n;
    int32_t c;
    int32_t h;
    int32_t w;
};

struct ReshapeParam {
    int32_t idim[4];
    int32_t itrans[4];
    int32_t odim[4];
    int32_t otrans[4];
};

struct Relu6Param {
    float threshold;
};

struct LeakyReluParam {
    float alpha;
};

struct HardSigmoidParam {
    float slope;
    float offset;
};

struct SwishParam {
    float beta;
};

struct HardSwishParam {
    float offset;
    float threshold;
    float scale;
};

struct ShuffleChannelParam {
    uint32_t group;
    uint32_t channel_per_group;
};

struct SplitParam {
    int32_t idim[4];
    int32_t axis;
    int32_t offset;
    int32_t num;
    int32_t v_;
    int32_t trans[4];
    int32_t vdim[4];
};

#pragma mark -

inline void xyzn2abcd_1(int xyzn[4], int abcd[4]) {
    abcd[0] = abcd[1] = abcd[2] = 0;
    abcd[3] = xyzn[0] * 4 + xyzn[3];
}
inline void xyzn2abcd_2(int xyzn[4], int abcd[4]) {
    abcd[0] = abcd[1] = 0;
    abcd[2] = xyzn[1];
    abcd[3] = xyzn[0] * 4 + xyzn[3];
}
inline void xyzn2abcd_3(int xyzn[4], int abcd[4]) {
    abcd[0] = 0;
    abcd[3] = xyzn[0];
    abcd[2] = xyzn[1];
    abcd[1] = xyzn[2] * 4 + xyzn[3];
}
inline void xyzn2abcd_4(int C, int xyzn[4], int abcd[4]) {
    abcd[2] = xyzn[0];
    abcd[1] = xyzn[1];
    uint t = xyzn[2] * 4 + xyzn[3];
    abcd[0] = t / C;
    abcd[3] = t % C;
}

inline void abcd2xyzn_1(int abcd[4], int xyzn[4]) {
    xyzn[1] = xyzn[2] = 0;
    xyzn[0] = abcd[3] / 4;
    xyzn[1] = abcd[3] % 4;
}
inline void abcd2xyzn_2(int abcd[4], int xyzn[4]) {
    xyzn[2] = 0;
    xyzn[1] = abcd[2];
    xyzn[0] = abcd[3] / 4;
    xyzn[3] = abcd[3] % 4;
}
inline void abcd2xyzn_3(int abcd[4], int xyzn[4]) {
    xyzn[0] = abcd[3];
    xyzn[1] = abcd[2];
    xyzn[2] = abcd[1] / 4;
    xyzn[3] = abcd[1] % 4;
}
inline void abcd2xyzn_4(int C, int abcd[4], int xyzn[4]) {
    xyzn[0] = abcd[2];
    xyzn[1] = abcd[1];
    uint t = abcd[0] * C + abcd[3];
    xyzn[2] = t / 4;
    xyzn[3] = t % 4;
}

inline void xyzn2abcd(int C, int xyzn[4], int abcd[4]) {
    abcd[2] = xyzn[0];
    abcd[1] = xyzn[1];
    uint t = xyzn[2] * 4 + xyzn[3];
    abcd[0] = t / C;
    abcd[3] = t % C;
}

inline void abcd2xyzn(int C, int abcd[4], int xyzn[4]) {
    xyzn[0] = abcd[2];
    xyzn[1] = abcd[1];
    uint t = abcd[0] * C + abcd[3];
    xyzn[2] = t / 4;
    xyzn[3] = t % 4;
}

inline int32_t abcd2index(int32_t dim[4], int32_t abcd[4]) {
    int32_t r = abcd[0];
    r = r * dim[1] + abcd[1];
    r = r * dim[2] + abcd[2];
    r = r * dim[3] + abcd[3];
    return r;
}

inline void index2abcd(int32_t dim[4], int32_t ind, int32_t abcd[4]) {
    abcd[3] = ind % dim[3];
    ind /= dim[3];
    abcd[2] = ind % dim[2];
    ind /= dim[2];
    abcd[1] = ind % dim[1];
    ind /= dim[1];
    abcd[0] = ind;
}

inline void trans(int32_t trans[4], int32_t ipos[4], int32_t opos[4]) {
    for (int i = 0; i < 4; i++) {
        opos[i] = ipos[trans[i]];
    }
}

inline void invtrans(int32_t trans[4], int32_t ipos[4], int32_t opos[4]) {
    for (int i = 0; i < 4; i++) {
        opos[trans[i]] = ipos[i];
    }
}

#pragma -

inline ftype4 activation(const ftype4 input, constant MetalActivationParam& param) {
    switch (param.activationType) {
        case NONE:
            return input;
        case RELU:
            return fmax(0, input);
        case RELU6:
            return fmin(fmax(input, 0.0), ftype(param.threshold));
        case PRELU:
            return input;
        case LEAKY_RELU:
            return fmax(input, ftype(param.alpha) * input);
        case HARD_SIGMOID:
            return fmax(0.0, fmin(1.0, ftype(param.slope) * input + ftype(param.offset)));
        case HARD_SWISH:
            return (fmin(ftype(param.threshold), fmax(0.0, input + ftype(param.offset)))) * input /
                   param.scale;
    }
}

#pragma -

inline half4 getBiasHalf(uint3 gid,
    constant ElementwiseAddParam& addParam,
    texture2d_array<half, access::sample> biasTexture) {
    half4 output;
    if (addParam.fast == 1) {
        output = biasTexture.read(gid.xy, gid.z);
    } else if (addParam.addByChannel == 1) {
        output = biasTexture.read(uint2(0, 0), gid.z);
    } else {
        int32_t x_xyzn[4] = {int32_t(gid.x), int32_t(gid.y), int32_t(gid.z), 0}, x_abcd[4],
                t_abcd[4];
        int32_t y_abcd[4] = {0, 0, 0, 0}, y_xyzn[4];
        int32_t xtrans[4] = {
            addParam.xtrans[0], addParam.xtrans[1], addParam.xtrans[2], addParam.xtrans[3]};
        int32_t ytrans[4] = {
            addParam.ytrans[0], addParam.ytrans[1], addParam.ytrans[2], addParam.ytrans[3]};
        int32_t yshift = 4 - addParam.ylen - addParam.axis;
        for (int n = 0; n < 4; n++) {
            x_xyzn[3] = n;
            xyzn2abcd(addParam.xdim[3], x_xyzn, x_abcd);
            invtrans(xtrans, x_abcd, t_abcd);
            for (int k = addParam.axis; k < (addParam.axis + addParam.ylen); k++) {
                y_abcd[yshift + k] = t_abcd[k];
            }
            trans(ytrans, y_abcd, t_abcd);
            abcd2xyzn(addParam.ydim[3], t_abcd, y_xyzn);
            output[n] = biasTexture.read(uint2(y_xyzn[0], y_xyzn[1]), y_xyzn[2])[y_xyzn[3]];
        }
    }
    return output;
}

inline ftype4 get_bias(uint3 gid,
    constant ElementwiseAddParam& addParam,
    texture2d_array<ftype, access::sample> biasTexture) {
    ftype4 output = ftype4(0.0);
    if (addParam.fast == 1) {
        output = biasTexture.read(gid.xy, gid.z);
    } else if (addParam.addByChannel == 1) {
        output = biasTexture.read(uint2(0, 0), gid.z);
    } else {
    }
    return output;
}
