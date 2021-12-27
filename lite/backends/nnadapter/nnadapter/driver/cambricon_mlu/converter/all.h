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

#ifndef __NNADAPTER_DRIVER_CAMBRICON_MLU_CONVERTER_ALL_H__  // NOLINT
#define __NNADAPTER_DRIVER_CAMBRICON_MLU_CONVERTER_ALL_H__

REGISTER_CONVERTER(ADAPTIVE_AVERAGE_POOL_2D, ConvertAdaptivePool2D)
REGISTER_CONVERTER(ADD, ConvertElementwise)
REGISTER_CONVERTER(AVERAGE_POOL_2D, ConvertPool2D)
REGISTER_CONVERTER(BATCH_NORMALIZATION, ConvertBatchNormalization)
REGISTER_CONVERTER(CAST, ConvertCast)
REGISTER_CONVERTER(CLIP, ConvertClip)
REGISTER_CONVERTER(CONCAT, ConvertConcat)
REGISTER_CONVERTER(CONV_2D, ConvertConv2D)
REGISTER_CONVERTER(CONV_2D_TRANSPOSE, ConvertConv2DTranspose)
REGISTER_CONVERTER(FILL, ConvertFill)
REGISTER_CONVERTER(FILL_LIKE, ConvertFillLike)
REGISTER_CONVERTER(FLATTEN, ConvertFlatten)
REGISTER_CONVERTER(FULLY_CONNECTED, ConvertFullyConnected)
REGISTER_CONVERTER(GATHER, ConvertGather)
REGISTER_CONVERTER(LEAKY_RELU, ConvertLeakyRelu)
REGISTER_CONVERTER(LOG, ConvertLog)
REGISTER_CONVERTER(LP_NORMALIZATION, ConvertLpNormalization)
REGISTER_CONVERTER(MAX_POOL_2D, ConvertPool2D)
REGISTER_CONVERTER(MUL, ConvertElementwise)
REGISTER_CONVERTER(PAD, ConvertPad)
REGISTER_CONVERTER(POW, ConvertPow)
REGISTER_CONVERTER(RANGE, ConvertRange)
REGISTER_CONVERTER(REDUCE_MEAN, ConvertReduce)
REGISTER_CONVERTER(REDUCE_SUM, ConvertReduce)
REGISTER_CONVERTER(RELU, ConvertActivations)
REGISTER_CONVERTER(RELU6, ConvertActivations)
REGISTER_CONVERTER(RESHAPE, ConvertReshape)
REGISTER_CONVERTER(RESIZE_LINEAR, ConvertResizeLinear)
REGISTER_CONVERTER(RESIZE_NEAREST, ConvertResizeNearest)
REGISTER_CONVERTER(SHAPE, ConvertShape)
REGISTER_CONVERTER(SIGMOID, ConvertActivations)
REGISTER_CONVERTER(SOFTMAX, ConvertSoftmax)
REGISTER_CONVERTER(SQUEEZE, ConvertSqueeze)
REGISTER_CONVERTER(SUB, ConvertElementwise)
REGISTER_CONVERTER(TANH, ConvertActivations)
REGISTER_CONVERTER(TRANSPOSE, ConvertTranspose)
REGISTER_CONVERTER(UNSQUEEZE, ConvertUnsqueeze)

#endif  // NOLINT
