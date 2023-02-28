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

#include "operation/resize_nearest.h"
#include "driver/eeasytech_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace eeasytech_npu {

int ConvertResizeNearest(Converter *converter, core::Operation *operation) {
  RESIZE_NEAREST_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK(!IsOperandWithDynamicShape(input_operand));

  // Convert to eeasynpu tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (input_tensor == nullptr) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto output_tensor = converter->ConvertOperand(output_operand);
  uint32_t scales[2];
  uint32_t sizes[2];
  std::shared_ptr<eeasy::nn::Tensor> shape_tensor = nullptr;
  std::shared_ptr<eeasy::nn::Tensor> scale_tensor = nullptr;
  if (shape_operand != nullptr) {
    float *shape_data = reinterpret_cast<float *>(shape_operand->buffer);
    scales[0] = shape_data[0] / input_operand->type.dimensions.data[2];
    scales[1] = shape_data[1] / input_operand->type.dimensions.data[3];
    sizes[0] = shape_data[0];
    sizes[1] = shape_data[1];
  } else {
    float *scales_data = reinterpret_cast<float *>(scales_operand->buffer);
    NNADAPTER_VLOG(5) << "scales_data[0]: " << scales_data[0];
    NNADAPTER_VLOG(5) << "scales_data[1]: " << scales_data[1];
    scales[0] = scales_data[0];
    scales[1] = scales_data[1];
    sizes[0] = output_operand->type.dimensions.data[2];
    sizes[1] = output_operand->type.dimensions.data[3];
  }
  NNADAPTER_VLOG(5) << "scales[0]: " << scales[0];
  NNADAPTER_VLOG(5) << "scales[1]: " << scales[1];
  NNADAPTER_VLOG(5) << "sizes[0]: " << sizes[0];
  NNADAPTER_VLOG(5) << "sizes[1]: " << sizes[1];
  eeasy::nn::UpsampleAttr attr;
  attr.scales.push_back(scales[0]);
  attr.scales.push_back(scales[1]);
  attr.sizes.push_back(sizes[0]);
  attr.sizes.push_back(sizes[1]);
  std::vector<std::shared_ptr<eeasy::nn::Tensor>> input_tensors = {
      input_tensor};
  std::vector<std::shared_ptr<eeasy::nn::Tensor>> output_tensors = {
      output_tensor};
  converter->AddOperator(
      eeasy::nn::OperatorType::UPSAMPLE, input_tensors, output_tensors, &attr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace eeasytech_npu
}  // namespace nnadapter
