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

#include <vector>
#include "fake_ddk/tensor.h"

namespace fake_ddk {

/* Supported operator types
 * <pre>
 * Operator is created by Graph::AddOperator(type, inputs, outputs, attrs, ...)
 * and each Operator has different inputs and outputs, also has different
 * additional attrs.
 * e.g.
 *   CONV2D  inputs: [input, filter, bias]  outputs: [output]  attrs: Conv2DAttr
 *   It means that CONV2D needs to set 3 inputs and 1 outputs, and fill the
 * Conv2DAttr structure.
 *   Sample code as follow:
 *     std::vector<fake_ddk::Tensor*> inputs, outputs;
 *     inputs.push_back(input);
 *     inputs.push_back(filter));
 *     inputs.push_back(bias);
 *     outputs.push_back(output);
 *     fake_ddk::Conv2DAttr attr;
 *     ...   // fill attr
 *     graph->AddOperator(fake_ddk::OperatorType::CONV2D, inputs, outputs,
 * reinterpret_cast<void*>(&attr));
 * </pre>
*/

typedef enum { FAKE_DDK_CONV2D = 0 } OperatorType;

typedef struct {
  /* Use POD(Plain Old Data) type otherwise it will cause serialization error */
  /* Pad type default value shall be AUTO */
  PadType pad_type;
  /* Pad top, bottom, left, right */
  int32_t pad[4];
  /* Stride height, width */
  int32_t stride[2];
  /* Dilation height, width */
  int32_t dilation[2];
  /* Group */
  int32_t group;
  /* Fuse type: FUSE_NONE, FUSE_RELU, FUSE_RELU1 and FUSE_RELU6 */
  FuseType fuse_type;
} Conv2DAttr;

typedef union {
  Conv2DAttr conv2d_attr;
  /* Add the structure of the attributes of other operators */
} OperatorAttr;

typedef struct {
  OperatorType type;
  OperatorAttr attr;
  /* Inputs of the operator */
  std::vector<Tensor*> input_tensors;
  /* Outputs of the operator */
  std::vector<Tensor*> output_tensors;
} Operator;

}  // namespace fake_ddk
