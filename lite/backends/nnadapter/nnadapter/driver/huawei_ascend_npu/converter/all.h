// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#ifndef __NNADAPTER_DRIVER_HUAWEI_ASCEND_NPU_CONVERTER_ALL_H__  // NOLINT
#define __NNADAPTER_DRIVER_HUAWEI_ASCEND_NPU_CONVERTER_ALL_H__

REGISTER_CONVERTER(ABS, ConvertUnaryActivations)
REGISTER_CONVERTER(ADAPTIVE_AVERAGE_POOL_2D, ConvertAdaptivePool2D)
REGISTER_CONVERTER(ADAPTIVE_MAX_POOL_2D, ConvertAdaptivePool2D)
REGISTER_CONVERTER(ADD, ConvertElementwise)
REGISTER_CONVERTER(ARG_MAX, ConvertArgMinMax)
REGISTER_CONVERTER(ARG_MIN, ConvertArgMinMax)
REGISTER_CONVERTER(ASSIGN, ConvertAssign)
REGISTER_CONVERTER(AVERAGE_POOL_2D, ConvertPool2D)
REGISTER_CONVERTER(BATCH_NORMALIZATION, ConvertBatchNormalization)
REGISTER_CONVERTER(CAST, ConvertCast)
REGISTER_CONVERTER(CLIP, ConvertClip)
REGISTER_CONVERTER(CONCAT, ConvertConcat)
REGISTER_CONVERTER(CONV_2D, ConvertConv2D)
REGISTER_CONVERTER(CONV_2D_TRANSPOSE, ConvertConv2DTranspose)
REGISTER_CONVERTER(CUM_SUM, ConvertCumSum)
REGISTER_CONVERTER(DEFORMABLE_CONV_2D, ConvertDeformableConv2d)
REGISTER_CONVERTER(DIV, ConvertElementwise)
REGISTER_CONVERTER(EQUAL, ConvertComparisons)
REGISTER_CONVERTER(EXP, ConvertUnaryActivations)
REGISTER_CONVERTER(EXPAND, ConvertExpand)
REGISTER_CONVERTER(FILL, ConvertFill)
REGISTER_CONVERTER(FULLY_CONNECTED, ConvertFullyConnected)
REGISTER_CONVERTER(GELU, ConvertGelu)
REGISTER_CONVERTER(GREATER, ConvertComparisons)
REGISTER_CONVERTER(GREATER_EQUAL, ConvertComparisons)
REGISTER_CONVERTER(HARD_SIGMOID, ConvertHardSigmoid)
REGISTER_CONVERTER(HARD_SWISH, ConvertHardSwish)
REGISTER_CONVERTER(LEAKY_RELU, ConvertLeakyRelu)
REGISTER_CONVERTER(LESS, ConvertComparisons)
REGISTER_CONVERTER(LESS_EQUAL, ConvertComparisons)
REGISTER_CONVERTER(LOG, ConvertUnaryActivations)
REGISTER_CONVERTER(LP_NORMALIZATION, ConvertLpNormalization)
REGISTER_CONVERTER(MAX, ConvertElementwise)
REGISTER_CONVERTER(MAX_POOL_2D, ConvertPool2D)
REGISTER_CONVERTER(MIN, ConvertElementwise)
REGISTER_CONVERTER(MUL, ConvertElementwise)
REGISTER_CONVERTER(NOT_EQUAL, ConvertComparisons)
REGISTER_CONVERTER(PAD, ConvertPad)
REGISTER_CONVERTER(POW, ConvertElementwise)
REGISTER_CONVERTER(PRELU, ConvertPRelu)
REGISTER_CONVERTER(RANGE, ConvertRange)
REGISTER_CONVERTER(REDUCE_MEAN, ConvertReduce)
REGISTER_CONVERTER(RELU, ConvertUnaryActivations)
REGISTER_CONVERTER(RELU6, ConvertUnaryActivations)
REGISTER_CONVERTER(RESHAPE, ConvertReshape)
REGISTER_CONVERTER(RESIZE_LINEAR, ConvertResizeLinear)
REGISTER_CONVERTER(RESIZE_NEAREST, ConvertResizeNearest)
REGISTER_CONVERTER(SHAPE, ConvertShape)
REGISTER_CONVERTER(SIGMOID, ConvertUnaryActivations)
REGISTER_CONVERTER(SLICE, ConvertSlice)
REGISTER_CONVERTER(SOFTMAX, ConvertSoftmax)
REGISTER_CONVERTER(SPLIT, ConvertSplit)
REGISTER_CONVERTER(SQUEEZE, ConvertSqueeze)
REGISTER_CONVERTER(STACK, ConvertStack)
REGISTER_CONVERTER(SUB, ConvertElementwise)
REGISTER_CONVERTER(SWISH, ConvertSwish)
REGISTER_CONVERTER(TANH, ConvertUnaryActivations)
REGISTER_CONVERTER(TOP_K, ConvertTopK)
REGISTER_CONVERTER(TRANSPOSE, ConvertTranspose)
REGISTER_CONVERTER(UNSQUEEZE, ConvertUnsqueeze)

#endif  // NOLINT
