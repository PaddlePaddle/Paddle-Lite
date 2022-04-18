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

#include "operation/elementwise.h"
#include "driver/nvidia_tensorrt/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace nvidia_tensorrt {

int ConvertElementwise(Converter* converter, core::Operation* operation) {
  ELEMENTWISE_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK_EQ(fuse_code, NNADAPTER_FUSED_NONE);

  // Convert to trt tensors and node
  auto input0_tensor = converter->GetMappedTensor(input0_operand);
  if (!input0_tensor) {
    // Tensorrt elementwise layer's input tensors must have the same number of
    // dimensions.
    auto dims = GetAlignedDims(input0_operand->type.dimensions,
                               input1_operand->type.dimensions);
    dims.erase(dims.begin());
    input0_tensor = converter->ConvertOperand(input0_operand, dims);
  }
  auto input1_tensor = converter->GetMappedTensor(input1_operand);
  if (!input1_tensor) {
    // Tensorrt elementwise layer's input tensors must have the same number of
    // dimensions.
    auto dims = GetAlignedDims(input1_operand->type.dimensions,
                               input0_operand->type.dimensions);
    dims.erase(dims.begin());
    input1_tensor = converter->ConvertOperand(input1_operand, dims);
  }
  // In order to solve the problem that the result is wrong after elementwise is
  // fused into scale
  if ((IsConstantOperand(input0_operand) &&
       !IsConstantOperand(input1_operand)) ||
      (!IsConstantOperand(input0_operand) &&
       IsConstantOperand(input1_operand))) {
    auto input0_dims_count = input0_operand->type.dimensions.count;
    auto input1_dims_count = input0_operand->type.dimensions.count;
    NNADAPTER_CHECK(input0_dims_count == input1_dims_count)
        << "The input dims count of elementwise should be equal. But "
           "input0_dims_count != input1_dims_count, "
        << input0_dims_count << " != " << input1_dims_count;
    int channel_axis = input0_dims_count > 1 ? 1 : 0;
    nvinfer1::IScaleLayer* scale_layer = nullptr;
    if (IsConstantOperand(input0_operand) &&
        !IsConstantOperand(input1_operand)) {
      int scale_weight_count =
          input0_operand->length /
          GetOperandPrecisionDataLength(input0_operand->type.precision);
      std::vector<float> zero_data(scale_weight_count, 0);
      auto offset_weight =
          converter->AddWeights(zero_data.data(), zero_data.size());
      auto power_weight =
          converter->AddWeights(zero_data.data(), zero_data.size());
      auto scale_weight = converter->OperandToWeights(input0_operand);
      scale_layer =
          converter->network()->addScaleNd(*input1_tensor,
                                           nvinfer1::ScaleMode::kELEMENTWISE,
                                           offset_weight,
                                           scale_weight,
                                           power_weight,
                                           channel_axis);
    } else if (!IsConstantOperand(input0_operand) &&
               IsConstantOperand(input1_operand)) {
      int scale_weight_count =
          input1_operand->length /
          GetOperandPrecisionDataLength(input1_operand->type.precision);
      std::vector<float> zero_data(scale_weight_count, 0);
      auto offset_weight =
          converter->AddWeights(zero_data.data(), zero_data.size());
      auto power_weight =
          converter->AddWeights(zero_data.data(), zero_data.size());
      auto scale_weight = converter->OperandToWeights(input1_operand);
      scale_layer =
          converter->network()->addScaleNd(*input0_tensor,
                                           nvinfer1::ScaleMode::kELEMENTWISE,
                                           offset_weight,
                                           scale_weight,
                                           power_weight,
                                           channel_axis);
    }
    auto output_tensor = scale_layer->getOutput(0);
    converter->UpdateTensorMap(output_operand, output_tensor);
    return NNADAPTER_NO_ERROR;
  }

  std::map<NNAdapterOperationType, nvinfer1::ElementWiseOperation>
      elementwise_type_map{
          {NNADAPTER_ADD, nvinfer1::ElementWiseOperation::kSUM},
          {NNADAPTER_MUL, nvinfer1::ElementWiseOperation::kPROD},
          {NNADAPTER_SUB, nvinfer1::ElementWiseOperation::kSUB},
          {NNADAPTER_DIV, nvinfer1::ElementWiseOperation::kDIV},
          {NNADAPTER_POW, nvinfer1::ElementWiseOperation::kPOW},
      };
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
