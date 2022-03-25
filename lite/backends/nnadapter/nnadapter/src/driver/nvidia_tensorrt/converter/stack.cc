// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "operation/stack.h"
#include "driver/nvidia_tensorrt/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace nvidia_tensorrt {

int ConvertStack(Converter* converter, core::Operation* operation) {
  STACK_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to trt tensors and node
  std::vector<nvinfer1::ITensor*> input_tensors;
  if (axis < input_dimensions_count) {
    for (int i = 0; i < input_count - 1; i++) {
      auto input_operand = input_operands[i];
      auto input_operator = converter->GetMappedTensor(input_operand);
      if (!input_operator) {
        input_operator = converter->ConvertOperand(input_operand);
      }
      input_tensors.push_back(input_operator);
    }
  } else {
    for (int i = 0; i < input_count - 1; i++) {
      nvinfer1::Dims reshape_dim;
      auto input_operand = input_operands[i];
      auto input_operator = converter->GetMappedTensor(input_operand);
      if (!input_operator) {
        input_operator = converter->ConvertOperand(input_operand);
      }
      reshape_dim.nbDims = input_dimensions_count + 1;
      reshape_dim.d[input_dimensions_count] = 1;
      auto reshape_layer = converter->network()->addShuffle(*input_operator);
      reshape_layer->setReshapeDimensions(reshape_dim);
      auto output_ = reshape_layer->getOutput(0);
      input_tensors.push_back(output_);
    }
  }
  auto stack_layer = converter->network()->addConcatenation(
      input_tensors.data(), input_count - 1);
  NNADAPTER_CHECK(stack_layer);
  stack_layer->setAxis(axis);
  auto stack_output_tensor = stack_layer->getOutput(0);
  nvinfer1::Dims reshape_dim;
  NNADAPTER_CHECK(!IsOperandWithDynamicShape(output_operand));
  reshape_dim.nbDims = output_operand->type.dimensions.count;
  for (int i = 0; i < reshape_dim.nbDims; i++) {
    reshape_dim.d[i] = output_operand->type.dimensions.data[i];
  }
  auto reshape_layer = converter->network()->addShuffle(*stack_output_tensor);
  reshape_layer->setReshapeDimensions(reshape_dim);
  auto output_tensor = reshape_layer->getOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
