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

#ifndef __NNADAPTER_DRIVER_ANDROID_NNAPI_CONVERTER_ALL_H__  // NOLINT
#define __NNADAPTER_DRIVER_ANDROID_NNAPI_CONVERTER_ALL_H__

REGISTER_CONVERTER(ADD, ValidateElementwise, ConvertElementwise)
REGISTER_CONVERTER(AVERAGE_POOL_2D, ValidatePool2D, ConvertPool2D)
REGISTER_CONVERTER(BATCH_NORMALIZATION,
                   ValidateBatchNormalization,
                   ConvertBatchNormalization)
REGISTER_CONVERTER(CONCAT, ValidateConcat, ConvertConcat)
REGISTER_CONVERTER(CONV_2D, ValidateConv2D, ConvertConv2D)
REGISTER_CONVERTER(CONV_2D_TRANSPOSE,
                   ValidateConv2DTranspose,
                   ConvertConv2DTranspose)
REGISTER_CONVERTER(DIV, ValidateElementwise, ConvertElementwise)
REGISTER_CONVERTER(FLATTEN, ValidateFlatten, ConvertFlatten)
REGISTER_CONVERTER(FULLY_CONNECTED,
                   ValidateFullyConnected,
                   ConvertFullyConnected)
REGISTER_CONVERTER(LEAKY_RELU, ValidateLeakyRelu, ConvertLeakyRelu)
REGISTER_CONVERTER(MAT_MUL, ValidateMatmul, ConvertMatmul)
REGISTER_CONVERTER(MAX_POOL_2D, ValidatePool2D, ConvertPool2D)
REGISTER_CONVERTER(MUL, ValidateElementwise, ConvertElementwise)
REGISTER_CONVERTER(RELU, ValidateUnaryActivations, ConvertUnaryActivations)
REGISTER_CONVERTER(RELU6, ValidateUnaryActivations, ConvertUnaryActivations)
REGISTER_CONVERTER(RESHAPE, ValidateReshape, ConvertReshape)
REGISTER_CONVERTER(RESIZE_LINEAR, ValidateResizeLinear, ConvertResizeLinear)
REGISTER_CONVERTER(RESIZE_NEAREST, ValidateResizeNearest, ConvertResizeNearest)
REGISTER_CONVERTER(SIGMOID, ValidateUnaryActivations, ConvertUnaryActivations)
REGISTER_CONVERTER(SOFTMAX, ValidateSoftmax, ConvertSoftmax)
REGISTER_CONVERTER(SUB, ValidateElementwise, ConvertElementwise)
REGISTER_CONVERTER(TANH, ValidateUnaryActivations, ConvertUnaryActivations)
REGISTER_CONVERTER(TRANSPOSE, ValidateTranspose, ConvertTranspose)
REGISTER_CONVERTER(UNSQUEEZE, ValidateUnsqueeze, ConvertUnsqueeze)

#endif  // NOLINT
