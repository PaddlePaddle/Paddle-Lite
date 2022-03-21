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

#include "operation/comparisons.h"
#include "driver/nvidia_tensorrt/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace nvidia_tensorrt {

int ConvertComparisons(Converter* converter, core::Operation* operation) {
  COMPARISONS_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to trt tensors and node
  auto input0_tensor = converter->GetMappedTensor(input0_operand);
  if (!input0_tensor) {
    input0_tensor = converter->ConvertOperand(input0_operand);
  }
  auto input1_tensor = converter->GetMappedTensor(input1_operand);
  if (!input1_tensor) {
    // Tensorrt elementwise layer's input tensors must have the same number of
    // dimensions.
    auto dims0_data = input0_operand->type.dimensions.data;
    auto dims0_count = input0_operand->type.dimensions.count;
    std::vector<int32_t> dims0(dims0_data, dims0_data + dims0_count);
    auto dims1_data = input1_operand->type.dimensions.data;
    auto dims1_count = input1_operand->type.dimensions.count;
    std::vector<int32_t> dims1(dims1_data, dims1_data + dims1_count);
    for (size_t i = 0; i < dims0.size(); i++) {
      if (dims1.size() < i + 1) {
        dims1.insert(dims1.begin(), 1);
        continue;
      }
      int data0 = dims0[dims0.size() - 1 - i];
      int data1 = dims1[dims1.size() - 1 - i];
      if (data0 != data1 && data0 != 1 && data1 != 1) {
        dims1.push_back(1);
      }
    }
    input1_tensor = converter->ConvertOperand(input1_operand, dims1);
  }
  std::map<NNAdapterOperationType, nvinfer1::ElementWiseOperation>
      elementwise_type_map{
          {NNADAPTER_EQUAL, nvinfer1::ElementWiseOperation::kEQUAL}};
  auto operation_type = operation->type;
  NNADAPTER_CHECK(elementwise_type_map.count(operation_type))
      << "Not support operation_type: "
      << OperationTypeToString(operation_type);
  auto elementwise_layer = converter->network()->addElementWise(
      *input0_tensor, *input1_tensor, elementwise_type_map.at(operation_type));
  NNADAPTER_CHECK(elementwise_layer);
  auto output_tensor = elementwise_layer->getOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
