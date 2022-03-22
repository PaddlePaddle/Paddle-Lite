// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <string>
#include "fakedevice/tensor.h"

namespace fake_ddk {
namespace nn {

/* Supported operator types
 * <pre>
 * Operator is created by Graph::AddOperator(type, inputs, outputs, attrs, ...)
 * and each Operator has different inputs and outputs, also has different
 * additional attrs.
 * e.g.
 *    CONV2D     inputs: [in, weight, bias]      outputs: [out]      attrs:
 * Conv2DAttr
 *    It means that CONV2D needs to set 3 inputs and 1 outputs, and fill the
 * Conv2DAttr structure.
 *    simple code as follow:
 *        std::vector<std::shared_ptr<fake_ddk::nn::Tensor>> inputs, outputs;
 *        inputs.push_back(in);
 *        inputs.push_back(weight);
 *        inputs.push_back(bias);
 *        outputs.push_back(out);
 *        fake_ddk::nn::Conv2DAttr attr;
 *        ...   // fill attr
 *        graph->AddOperator(fake_ddk::nn::OperatorType::CONV2D, inputs,
 * outputs, (void*)&attr);
 * </pre>
 */

typedef enum { FAKE_DEVICE_CONV2D, FAKE_DEVICE_RELU } fakedevice_nn_op_t;
typedef fakedevice_nn_op_t OperatorType;

typedef struct _fakedevice_nn_conv2d_param {
  uint32_t ksize[2];
  uint32_t stride[2];
  /* Pad left, right, top, bottom */
  uint32_t pad[4];
  /* Pad type default value shall be AUTO */
  PadType pad_type;
  uint32_t weights;
  uint32_t group;
  uint32_t dilation[2];
  int32_t multiplier;
  bool has_relu;
} fakedevice_nn_conv2d_param;
typedef fakedevice_nn_conv2d_param Conv2DAttr;

typedef union _fakedevice_nn_param {
  fakedevice_nn_conv2d_param conv2d_param;
} fakedevice_nn_param_t;
}  // namespace nn
}  // namespace fake_ddk
