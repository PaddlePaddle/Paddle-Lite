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

#ifndef __NNADAPTER_OPERATION_ALL_H__  // NOLINT
#define __NNADAPTER_OPERATION_ALL_H__

REGISTER_OPERATION(ABS,
                   ValidateUnaryActivations,
                   PrepareUnaryActivations,
                   ExecuteUnaryActivations)
REGISTER_OPERATION(ADAPTIVE_AVERAGE_POOL_2D,
                   ValidateAdaptivePool2D,
                   PrepareAdaptivePool2D,
                   ExecuteAdaptivePool2D)
REGISTER_OPERATION(ADAPTIVE_MAX_POOL_2D,
                   ValidateAdaptivePool2D,
                   PrepareAdaptivePool2D,
                   ExecuteAdaptivePool2D)
REGISTER_OPERATION(ADD,
                   ValidateElementwise,
                   PrepareElementwise,
                   ExecuteElementwise)
REGISTER_OPERATION(AND,
                   ValidateBinaryLogicalOp,
                   PrepareBinaryLogicalOp,
                   ExecuteBinaryLogicalOp)
REGISTER_OPERATION(ASSIGN, ValidateAssign, PrepareAssign, ExecuteAssign)
REGISTER_OPERATION(ARG_MAX,
                   ValidateArgMinMax,
                   PrepareArgMinMax,
                   ExecuteArgMinMax)
REGISTER_OPERATION(ARG_MIN,
                   ValidateArgMinMax,
                   PrepareArgMinMax,
                   ExecuteArgMinMax)
REGISTER_OPERATION(AVERAGE_POOL_2D,
                   ValidatePool2D,
                   PreparePool2D,
                   ExecutePool2D)
REGISTER_OPERATION(BATCH_NORMALIZATION,
                   ValidateBatchNormalization,
                   PrepareBatchNormalization,
                   ExecuteBatchNormalization)
REGISTER_OPERATION(CAST, ValidateCast, PrepareCast, ExecuteCast)
REGISTER_OPERATION(CLIP, ValidateClip, PrepareClip, ExecuteClip)
REGISTER_OPERATION(CHANNEL_SHUFFLE,
                   ValidateChannelShuffle,
                   PrepareChannelShuffle,
                   ExecuteChannelShuffle)
REGISTER_OPERATION(CONCAT, ValidateConcat, PrepareConcat, ExecuteConcat)
REGISTER_OPERATION(CONV_2D, ValidateConv2D, PrepareConv2D, ExecuteConv2D)
REGISTER_OPERATION(CONV_2D_TRANSPOSE,
                   ValidateConv2DTranspose,
                   PrepareConv2DTranspose,
                   ExecuteConv2DTranspose)
REGISTER_OPERATION(CUM_SUM, ValidateCumSum, PrepareCumSum, ExecuteCumSum)
REGISTER_OPERATION(DEFORMABLE_CONV_2D,
                   ValidateDeformableConv2D,
                   PrepareDeformableConv2D,
                   ExecuteDeformableConv2D)
REGISTER_OPERATION(DEQUANTIZE,
                   ValidateDequantize,
                   PrepareDequantize,
                   ExecuteDequantize)
REGISTER_OPERATION(DIV,
                   ValidateElementwise,
                   PrepareElementwise,
                   ExecuteElementwise)
REGISTER_OPERATION(EQUAL,
                   ValidateComparisons,
                   PrepareComparisons,
                   ExecuteComparisons)
REGISTER_OPERATION(EXP,
                   ValidateUnaryActivations,
                   PrepareUnaryActivations,
                   ExecuteUnaryActivations)
REGISTER_OPERATION(FILL, ValidateFill, PrepareFill, ExecuteFill)
REGISTER_OPERATION(FILL_LIKE,
                   ValidateFillLike,
                   PrepareFillLike,
                   ExecuteFillLike)
REGISTER_OPERATION(FLATTEN, ValidateFlatten, PrepareFlatten, ExecuteFlatten)
REGISTER_OPERATION(FLOOR,
                   ValidateUnaryActivations,
                   PrepareUnaryActivations,
                   ExecuteUnaryActivations)
REGISTER_OPERATION(FULLY_CONNECTED,
                   ValidateFullyConnected,
                   PrepareFullyConnected,
                   ExecuteFullyConnected)
REGISTER_OPERATION(GATHER, ValidateGather, PrepareGather, ExecuteGather)
REGISTER_OPERATION(GELU, ValidateGelu, PrepareGelu, ExecuteGelu)
REGISTER_OPERATION(GREATER,
                   ValidateComparisons,
                   PrepareComparisons,
                   ExecuteComparisons)
REGISTER_OPERATION(GREATER_EQUAL,
                   ValidateComparisons,
                   PrepareComparisons,
                   ExecuteComparisons)
REGISTER_OPERATION(GRID_SAMPLE,
                   ValidateGridSample,
                   PrepareGridSample,
                   ExecuteGridSample)
REGISTER_OPERATION(GROUP_NORMALIZATION,
                   ValidateGroupNormalization,
                   PrepareGroupNormalization,
                   ExecuteGroupNormalization)
REGISTER_OPERATION(HARD_SIGMOID,
                   ValidateHardSigmoidSwish,
                   PrepareHardSigmoidSwish,
                   ExecuteHardSigmoidSwish)
REGISTER_OPERATION(HARD_SWISH,
                   ValidateHardSigmoidSwish,
                   PrepareHardSigmoidSwish,
                   ExecuteHardSigmoidSwish)
REGISTER_OPERATION(EXPAND, ValidateExpand, PrepareExpand, ExecuteExpand)
REGISTER_OPERATION(INSTANCE_NORMALIZATION,
                   ValidateInstanceNormalization,
                   PrepareInstanceNormalization,
                   ExecuteInstanceNormalization)
REGISTER_OPERATION(LAYER_NORMALIZATION,
                   ValidateLayerNormalization,
                   PrepareLayerNormalization,
                   ExecuteLayerNormalization)
REGISTER_OPERATION(LEAKY_RELU,
                   ValidateLeakyRelu,
                   PrepareLeakyRelu,
                   ExecuteLeakyRelu)
REGISTER_OPERATION(LESS,
                   ValidateComparisons,
                   PrepareComparisons,
                   ExecuteComparisons)
REGISTER_OPERATION(LESS_EQUAL,
                   ValidateComparisons,
                   PrepareComparisons,
                   ExecuteComparisons)
REGISTER_OPERATION(LOG,
                   ValidateUnaryActivations,
                   PrepareUnaryActivations,
                   ExecuteUnaryActivations)
REGISTER_OPERATION(LOG_SOFTMAX,
                   ValidateLogSoftmax,
                   PrepareLogSoftmax,
                   ExecuteLogSoftmax)
REGISTER_OPERATION(LP_NORMALIZATION,
                   ValidateLpNormalization,
                   PrepareLpNormalization,
                   ExecuteLpNormalization)
REGISTER_OPERATION(MAT_MUL, ValidateMatMul, PrepareMatMul, ExecuteMatMul)
REGISTER_OPERATION(MAX,
                   ValidateElementwise,
                   PrepareElementwise,
                   ExecuteElementwise)
REGISTER_OPERATION(MAX_POOL_2D, ValidatePool2D, PreparePool2D, ExecutePool2D)
REGISTER_OPERATION(MESHGRID, ValidateMeshgrid, PrepareMeshgrid, ExecuteMeshgrid)
REGISTER_OPERATION(MIN,
                   ValidateElementwise,
                   PrepareElementwise,
                   ExecuteElementwise)
REGISTER_OPERATION(MUL,
                   ValidateElementwise,
                   PrepareElementwise,
                   ExecuteElementwise)
REGISTER_OPERATION(NOT,
                   ValidateUnaryLogicalOp,
                   PrepareUnaryLogicalOp,
                   ExecuteUnaryLogicalOp)
REGISTER_OPERATION(NOT_EQUAL,
                   ValidateComparisons,
                   PrepareComparisons,
                   ExecuteComparisons)
REGISTER_OPERATION(PAD, ValidatePad, PreparePad, ExecutePad)
REGISTER_OPERATION(POW,
                   ValidateElementwise,
                   PrepareElementwise,
                   ExecuteElementwise)
REGISTER_OPERATION(PRELU, ValidatePRelu, PreparePRelu, ExecutePRelu)
REGISTER_OPERATION(PRIOR_BOX,
                   ValidatePriorBox,
                   PreparePriorBox,
                   ExecutePriorBox)
REGISTER_OPERATION(QUANTIZE, ValidateQuantize, PrepareQuantize, ExecuteQuantize)
REGISTER_OPERATION(RANGE, ValidateRange, PrepareRange, ExecuteRange)
REGISTER_OPERATION(REDUCE_MEAN, ValidateReduce, PrepareReduce, ExecuteReduce)
REGISTER_OPERATION(REDUCE_MAX, ValidateReduce, PrepareReduce, ExecuteReduce)
REGISTER_OPERATION(REDUCE_SUM, ValidateReduce, PrepareReduce, ExecuteReduce)
REGISTER_OPERATION(RELU,
                   ValidateUnaryActivations,
                   PrepareUnaryActivations,
                   ExecuteUnaryActivations)
REGISTER_OPERATION(RELU6,
                   ValidateUnaryActivations,
                   PrepareUnaryActivations,
                   ExecuteUnaryActivations)
REGISTER_OPERATION(RESHAPE, ValidateReshape, PrepareReshape, ExecuteReshape)
REGISTER_OPERATION(RESIZE_LINEAR, ValidateResize, PrepareResize, ExecuteResize)
REGISTER_OPERATION(RESIZE_NEAREST, ValidateResize, PrepareResize, ExecuteResize)
REGISTER_OPERATION(ROI_ALIGN,
                   ValidateRoiAlign,
                   PrepareRoiAlign,
                   ExecuteRoiAlign)
REGISTER_OPERATION(SHAPE, ValidateShape, PrepareShape, ExecuteShape)
REGISTER_OPERATION(SIGMOID,
                   ValidateUnaryActivations,
                   PrepareUnaryActivations,
                   ExecuteUnaryActivations)
REGISTER_OPERATION(SLICE, ValidateSlice, PrepareSlice, ExecuteSlice)
REGISTER_OPERATION(SOFTMAX, ValidateSoftmax, PrepareSoftmax, ExecuteSoftmax)
REGISTER_OPERATION(SOFTPLUS, ValidateSoftplus, PrepareSoftplus, ExecuteSoftplus)
REGISTER_OPERATION(SPLIT, ValidateSplit, PrepareSplit, ExecuteSplit)
REGISTER_OPERATION(SQUARE,
                   ValidateUnaryActivations,
                   PrepareUnaryActivations,
                   ExecuteUnaryActivations)
REGISTER_OPERATION(SQUEEZE, ValidateSqueeze, PrepareSqueeze, ExecuteSqueeze)
REGISTER_OPERATION(STACK, ValidateStack, PrepareStack, ExecuteStack)
REGISTER_OPERATION(SUB,
                   ValidateElementwise,
                   PrepareElementwise,
                   ExecuteElementwise)
REGISTER_OPERATION(SUM, ValidateSum, PrepareSum, ExecuteSum)
REGISTER_OPERATION(SWISH,
                   ValidateUnaryActivations,
                   PrepareUnaryActivations,
                   ExecuteUnaryActivations)
REGISTER_OPERATION(TANH,
                   ValidateUnaryActivations,
                   PrepareUnaryActivations,
                   ExecuteUnaryActivations)
REGISTER_OPERATION(TILE, ValidateTile, PrepareTile, ExecuteTile)
REGISTER_OPERATION(TOP_K, ValidateTopK, PrepareTopK, ExecuteTopK)
REGISTER_OPERATION(TRANSPOSE,
                   ValidateTranspose,
                   PrepareTranspose,
                   ExecuteTranspose)
REGISTER_OPERATION(UNSQUEEZE,
                   ValidateUnsqueeze,
                   PrepareUnsqueeze,
                   ExecuteUnsqueeze)
REGISTER_OPERATION(UNSTACK, ValidateUnstack, PrepareUnstack, ExecuteUnstack)
REGISTER_OPERATION(WHERE, ValidateWhere, PrepareWhere, ExecuteWhere)
REGISTER_OPERATION(YOLO_BOX, ValidateYoloBox, PrepareYoloBox, ExecuteYoloBox)
REGISTER_OPERATION(NON_MAX_SUPPRESSION,
                   ValidateNonMaxSuppression,
                   PrepareNonMaxSuppression,
                   ExecuteNonMaxSuppression)

#endif  // NOLINT
