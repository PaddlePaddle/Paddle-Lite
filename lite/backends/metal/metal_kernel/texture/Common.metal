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
    abcd[3] = ind % dim[3]; ind /= dim[3];
    abcd[2] = ind % dim[2]; ind /= dim[2];
    abcd[1] = ind % dim[1]; ind /= dim[1];
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

struct ElementwiseAddParam {
    int32_t fast;
    int32_t addByChannel;
    int32_t axis;
    int32_t ylen;
    int32_t xdim[4];
    int32_t xtrans[4];
    int32_t ydim[4];
    int32_t ytrans[4];
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

inline half4 getBiasHalf(uint3 gid, constant ElementwiseAddParam &addParam, texture2d_array<half, access::sample> biasTexture) {
    half4 output;
    if (addParam.fast == 1) {
        output = biasTexture.read(gid.xy, gid.z);
    } else if (addParam.addByChannel == 1) {
        output = biasTexture.read(uint2(gid.z, 0), 0);
    } else {
        int32_t x_xyzn[4] = {int32_t(gid.x), int32_t(gid.y), int32_t(gid.z), 0}, x_abcd[4], t_abcd[4];
        int32_t y_abcd[4] = {0, 0, 0, 0}, y_xyzn[4];
        int32_t xtrans[4] = {addParam.xtrans[0], addParam.xtrans[1], addParam.xtrans[2], addParam.xtrans[3]};
        int32_t ytrans[4] = {addParam.ytrans[0], addParam.ytrans[1], addParam.ytrans[2], addParam.ytrans[3]};
        int32_t yshift = 4 - addParam.ylen - addParam.axis;
        for (int n = 0; n < 4; n++) {
            x_xyzn[3] = n;
            xyzn2abcd(addParam.xdim[3], x_xyzn, x_abcd);
            invtrans(xtrans, x_abcd, t_abcd);
            for (int k = addParam.axis; k < (addParam.axis + addParam.ylen); k++) {
                y_abcd[yshift+k] = t_abcd[k];
            }
            trans(ytrans, y_abcd, t_abcd);
            abcd2xyzn(addParam.ydim[3], t_abcd, y_xyzn);
            output[n] = biasTexture.read(uint2(y_xyzn[0], y_xyzn[1]), y_xyzn[2])[y_xyzn[3]];
        }
    }
    return output;
}

inline float4 getBias(uint3 gid, constant ElementwiseAddParam &addParam, texture2d_array<float, access::sample> biasTexture) {
    float4 output;
    if (addParam.fast == 1) {
        output = float4(biasTexture.read(gid.xy, gid.z));
    } else if (addParam.addByChannel == 1) {
        output = float4(biasTexture.read(uint2(gid.z, 0), 0));
    } else {
        int32_t x_xyzn[4] = {int32_t(gid.x), int32_t(gid.y), int32_t(gid.z), 0}, x_abcd[4], t_abcd[4];
        int32_t y_abcd[4] = {0, 0, 0, 0}, y_xyzn[4];
        int32_t xtrans[4] = {addParam.xtrans[0], addParam.xtrans[1], addParam.xtrans[2], addParam.xtrans[3]};
        int32_t ytrans[4] = {addParam.ytrans[0], addParam.ytrans[1], addParam.ytrans[2], addParam.ytrans[3]};
        int32_t yshift = 4 - addParam.ylen - addParam.axis;
        for (int n = 0; n < 4; n++) {
            x_xyzn[3] = n;
            xyzn2abcd(addParam.xdim[3], x_xyzn, x_abcd);
            invtrans(xtrans, x_abcd, t_abcd);
            for (int k = addParam.axis; k < (addParam.axis + addParam.ylen); k++) {
                y_abcd[yshift+k] = t_abcd[k];
            }
            trans(ytrans, y_abcd, t_abcd);
            abcd2xyzn(addParam.ydim[3], t_abcd, y_xyzn);
            output[n] = biasTexture.read(uint2(y_xyzn[0], y_xyzn[1]), y_xyzn[2])[y_xyzn[3]];
        }
    }
    return output;
}
