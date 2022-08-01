// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef LITE_KERNELS_METAL_IMAGE_OP_METAL_PARAMS_H_
#define LITE_KERNELS_METAL_IMAGE_OP_METAL_PARAMS_H_

#include <cstdint>

struct ElementwiseAddMetalParam {
    int fast;
    int addByChannel;
    int axis;
    int ylen;
    int xdim[4];
    int xtrans[4];
    int ydim[4];
    int ytrans[4];
    int ByNum;
    int ByHW;
    int ByW;
    int arithmetic_type;
};

struct ActivationMetalParam {
    uint16_t activationType;
    float threshold;  // RELU6
    float alpha;      // LEAKY_RELU
    float offset;     // HARD_SIGMOID
    float slope;      // HARD_SIGMOID
    float scale;      // HARD_SWISH
};

struct MetalConvParam {
    int16_t offsetX;
    int16_t offsetY;
    int16_t offsetZ;
    uint16_t strideX;
    uint16_t strideY;
    uint16_t dilationX;
    uint16_t dilationY;
    uint16_t groups;
    uint16_t iC;
    uint16_t fC;
    uint16_t oC;
    uint16_t hasAddOp;
    uint16_t hasReluOp;
    ElementwiseAddMetalParam addParam;
    ActivationMetalParam activationParam;
};

struct SoftmaxMetalParam {
    int N;
    int K;
};

struct SoftmaxMetalParam2 {
    int N;
    int C;
    int H;
    int W;
};

struct ScaleMetalParam {
    float scale;
    float abias;
    ActivationMetalParam activationParam;
};

struct ReshapeMetalParam {
    int idim[4];
    int itrans[4];
    int odim[4];
    int otrans[4];
};

struct Relu6MetalParam {
    float threshold;
};

struct LeakyReluMetalParam {
    float alpha;
};

struct PoolMetalParam {
    int ksizeX;
    int ksizeY;
    int strideX;
    int strideY;
    int paddingX;
    int paddingY;
    int poolType;
    int exclusive;
};

struct MulMetalParam {};

struct MatmulMetalParam {
    bool transposeX;
    bool transposeY;
    bool broadcast;
};

struct FCMetalParam {
    int N;
    int K;
};

struct DropoutMetalParam {
    float scale;
};

struct DepthwiseConv2dMetalParam {
    int16_t offsetX;
    int16_t offsetY;
    int16_t offsetZ;
    uint16_t strideX;
    uint16_t strideY;
    uint16_t dilationX;
    uint16_t dilationY;
    uint16_t groups;
    uint16_t iC;
    uint16_t fC;
    uint16_t oC;
    uint16_t hasAddOp;
    uint16_t hasReluOp;
    ElementwiseAddMetalParam addParam;
};

struct ConcatMetalParam {
    int odim[4];
    int axis;
    int offset;
    int num;
    int v_;
    int trans[4];
    int vdim[6];
};

struct BilinearInterPMetalParam {
    float ratio_h;
    float ratio_w;
    float align_delta;
};

struct NearestInterpMetalParam {
    float ratioH;
    float ratioW;
    float alignDelta;
};

struct PixelShuffleMetalParam {
    int upscale_factor;
};

struct ShuffleChannelMetalParam {
    uint32_t group;
    uint32_t channel_per_group;
};

struct LrnMetalParam {
    int n;
    int channelN;
    float k;
    float alpha;
    float beta;
};

struct InstanceNormReluMetalParam {
    uint16_t hasReluOp;
};

struct HardSigmoidMetalParam {
    float slope;
    float offset;
};

struct SwishMetalParam {
    float beta;
};

struct HardSwishMetalParam {
    float offset;
    float threshold;
    float scale;
};

struct ExpandMetalParam {
    uint16_t fast;
    uint16_t c;
    uint16_t h;
    uint16_t w;
};

struct ElementwiseMetalParam {
    int byChannel;
};

struct TransposeMetalParam {
    int idim[4];
    int itrans[4];
    int odim[4];
    int otrans[4];
    int axis[4];
};

struct PriorBoxMetalParam {
    float offset;
    float stepWidth;
    float stepHeight;
    float minSize;
    float maxSize;
    float imageWidth;
    float imageHeight;
    bool clip;
    uint32_t numPriors;
    uint32_t aspecRatiosSize;
    uint32_t minSizeSize;
    uint32_t maxSizeSize;
};

struct SplitMetalParam {
    int idim[4];
    int axis;
    int offset;
    int num;
    int v_;
    int trans[4];
    int vdim[4];
};

struct ConvTransposeAddMetalParam {
    uint16_t kernelW;
    uint16_t kernelH;
    uint16_t strideX;
    uint16_t strideY;
    uint16_t paddingX;
    uint16_t paddingY;
    uint16_t dilationX;
    uint16_t dilationY;
    uint16_t groups;
    uint16_t iC;
    uint16_t fC;
    uint16_t oC;
    uint16_t hasAddOp;
    ElementwiseAddMetalParam addParam;
    ActivationMetalParam activationParam;
};

struct SliceMetalParam {
    int iW;
    int iH;
    int oW;
    int oH;
    int isize;
    int osize;
    int oarraysize;
    int start[4];
    int endC;
};

struct FeedMetalParam {
    int isize;
    int idim[4];
};

struct FetchMetalParam {
    int isize;
    int idim[4];
};

struct Pad2dParam {
    uint16_t paddingTop;
    uint16_t paddingBottom;
    uint16_t paddingLeft;
    uint16_t paddingRight;
    float padValue;
    uint16_t mode;
};

struct YoloBoxMetalParam {
    int imgH;
    int imgW;
    int xN;
    int xC;
    int xH;
    int xW;
    int xStride;
    int xSize;
    int boxNum;
    int anchorNum;
    int anchorStride;
    int classNum;
    int clipBBox;
    float confThresh;
    float scale;
    float bias;
};

struct CompareMetalParam {
    int compareType;
};

struct CastMetalParam {
    int inType;
    int outType;
};

struct ArgMetalParam {
    int orank;
};

#endif  // LITE_KERNELS_METAL_IMAGE_OP_METAL_PARAMS_H_
