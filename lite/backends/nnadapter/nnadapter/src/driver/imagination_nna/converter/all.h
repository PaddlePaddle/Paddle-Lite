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

#ifndef __NNADAPTER_DRIVER_IMAGINATION_NNA_CONVERTER_ALL_H__  // NOLINT
#define __NNADAPTER_DRIVER_IMAGINATION_NNA_CONVERTER_ALL_H__

REGISTER_CONVERTER(AVERAGE_POOL_2D, ConvertPool2D)
REGISTER_CONVERTER(CONV_2D, ConvertConv2D)
REGISTER_CONVERTER(FULLY_CONNECTED, ConvertFullyConnected)
REGISTER_CONVERTER(MAX_POOL_2D, ConvertPool2D)
REGISTER_CONVERTER(RELU, ConvertUnaryActivations)
REGISTER_CONVERTER(RELU6, ConvertUnaryActivations)
REGISTER_CONVERTER(SOFTMAX, ConvertSoftmax)
REGISTER_CONVERTER(MAT_MUL, ConvertMatMul)
REGISTER_CONVERTER(ADD, ConvertElementwise)
REGISTER_CONVERTER(SUB, ConvertElementwise)
REGISTER_CONVERTER(MUL, ConvertElementwise)
REGISTER_CONVERTER(DIV, ConvertElementwise)
REGISTER_CONVERTER(MAX, ConvertElementwise)
REGISTER_CONVERTER(MIN, ConvertElementwise)
REGISTER_CONVERTER(RESHAPE, ConvertReshape)

#endif  // NOLINT
