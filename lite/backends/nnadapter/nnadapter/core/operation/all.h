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

REGISTER_OPERATION(CONV_2D, PrepareConv2D)
REGISTER_OPERATION(SOFTMAX, PrepareSoftmax)
REGISTER_OPERATION(SHAPE, PrepareShape)
REGISTER_OPERATION(FILL, PrepareFill)
REGISTER_OPERATION(INSTANCE_NORMALIZATION, PrepareInstanceNormalization)
REGISTER_OPERATION(ABS, PrepareUnaryActivations)
REGISTER_OPERATION(EXP, PrepareUnaryActivations)
REGISTER_OPERATION(LOG, PrepareUnaryActivations)
REGISTER_OPERATION(RELU, PrepareUnaryActivations)
REGISTER_OPERATION(RELU6, PrepareUnaryActivations)
REGISTER_OPERATION(SIGMOID, PrepareUnaryActivations)
REGISTER_OPERATION(TANH, PrepareUnaryActivations)
REGISTER_OPERATION(CUM_SUM, PrepareCumSum)

#endif  // NOLINT
