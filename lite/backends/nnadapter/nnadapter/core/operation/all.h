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

#ifndef __NNADAPTER_CORE_OPERATION_ALL_H__  // NOLINT
#define __NNADAPTER_CORE_OPERATION_ALL_H__

REGISTER_OPERATION(ABS, PrepareUnaryActivations)
REGISTER_OPERATION(ADAPTIVE_AVERAGE_POOL_2D, PrepareAdaptivePool2D)
REGISTER_OPERATION(ADAPTIVE_MAX_POOL_2D, PrepareAdaptivePool2D)
REGISTER_OPERATION(ADD, PrepareElementwise)
REGISTER_OPERATION(ASSIGN, PrepareAssign)
REGISTER_OPERATION(ARG_MAX, PrepareArgMinMax)
REGISTER_OPERATION(ARG_MIN, PrepareArgMinMax)
REGISTER_OPERATION(AVERAGE_POOL_2D, PreparePool2D)
REGISTER_OPERATION(BATCH_NORMALIZATION, PrepareBatchNormalization)
REGISTER_OPERATION(CLIP, PrepareClip)
REGISTER_OPERATION(CONCAT, PrepareConcat)
REGISTER_OPERATION(CONV_2D, PrepareConv2D)
REGISTER_OPERATION(CONV_2D_TRANSPOSE, PrepareConv2DTranspose)
REGISTER_OPERATION(CUM_SUM, PrepareCumSum)
REGISTER_OPERATION(DEFORMABLE_CONV_2D, PrepareDeformableConv2D)
REGISTER_OPERATION(DEQUANTIZE, PrepareDequantize)
REGISTER_OPERATION(DIV, PrepareElementwise)
REGISTER_OPERATION(EQUAL, PrepareComparisons)
REGISTER_OPERATION(EXP, PrepareUnaryActivations)
REGISTER_OPERATION(FILL, PrepareFill)
REGISTER_OPERATION(FLATTEN, PrepareFlatten)
REGISTER_OPERATION(GATHER, PrepareGather)
REGISTER_OPERATION(GELU, PrepareGelu)
REGISTER_OPERATION(GREATER, PrepareComparisons)
REGISTER_OPERATION(GREATER_EQUAL, PrepareComparisons)
REGISTER_OPERATION(HARD_SIGMOID, PrepareHardSigmoidSwish)
REGISTER_OPERATION(HARD_SWISH, PrepareHardSigmoidSwish)
REGISTER_OPERATION(EXPAND, PrepareExpand)
REGISTER_OPERATION(INSTANCE_NORMALIZATION, PrepareInstanceNormalization)
REGISTER_OPERATION(LAYER_NORMALIZATION, PrepareLayerNormalization)
REGISTER_OPERATION(LEAKY_RELU, PrepareLeakyRelu)
REGISTER_OPERATION(LESS, PrepareComparisons)
REGISTER_OPERATION(LESS_EQUAL, PrepareComparisons)
REGISTER_OPERATION(LOG, PrepareUnaryActivations)
REGISTER_OPERATION(MAT_MUL, PrepareMatMul)
REGISTER_OPERATION(MAX, PrepareElementwise)
REGISTER_OPERATION(MAX_POOL_2D, PreparePool2D)
REGISTER_OPERATION(MIN, PrepareElementwise)
REGISTER_OPERATION(MUL, PrepareElementwise)
REGISTER_OPERATION(NOT_EQUAL, PrepareComparisons)
REGISTER_OPERATION(POW, PrepareElementwise)
REGISTER_OPERATION(PRELU, PreparePRelu)
REGISTER_OPERATION(QUANTIZE, PrepareQuantize)
REGISTER_OPERATION(RANGE, PrepareRange)
REGISTER_OPERATION(REDUCE_MEAN, PrepareReduce)
REGISTER_OPERATION(RELU, PrepareUnaryActivations)
REGISTER_OPERATION(RELU6, PrepareUnaryActivations)
REGISTER_OPERATION(RESHAPE, PrepareReshape)
REGISTER_OPERATION(RESIZE_LINEAR, PrepareResize)
REGISTER_OPERATION(RESIZE_NEAREST, PrepareResize)
REGISTER_OPERATION(SHAPE, PrepareShape)
REGISTER_OPERATION(SIGMOID, PrepareUnaryActivations)
REGISTER_OPERATION(SLICE, PrepareSlice)
REGISTER_OPERATION(SOFTMAX, PrepareSoftmax)
REGISTER_OPERATION(SPLIT, PrepareSplit)
REGISTER_OPERATION(SQUEEZE, PrepareSqueeze)
REGISTER_OPERATION(STACK, PrepareStack)
REGISTER_OPERATION(SUB, PrepareElementwise)
REGISTER_OPERATION(SWISH, PrepareUnaryActivations)
REGISTER_OPERATION(TANH, PrepareUnaryActivations)
REGISTER_OPERATION(TOP_K, PrepareTopK)
REGISTER_OPERATION(TRANSPOSE, PrepareTranspose)
REGISTER_OPERATION(UNSQUEEZE, PrepareUnsqueeze)

#endif  // NOLINT
