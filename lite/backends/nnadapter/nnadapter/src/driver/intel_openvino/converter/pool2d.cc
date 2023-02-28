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

#include "operation/pool2d.h"
#include "driver/intel_openvino/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace intel_openvino {

int ConvertPool2D(Converter* converter, core::Operation* operation) {
  POOL_2D_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert operand to OpenVINO tensor
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto ov_auto_pad = ConvertToOVPadType(auto_pad);
  auto ov_pads_begin = Shape({static_cast<size_t>(pad_height_top),
                              static_cast<size_t>(pad_width_left)});
  auto ov_pads_end = Shape({static_cast<size_t>(pad_height_bottom),
                            static_cast<size_t>(pad_width_right)});
  auto ov_strides = ov::Strides(
      {static_cast<size_t>(stride_height), static_cast<size_t>(stride_width)});
  auto ov_kernel = Shape(
      {static_cast<size_t>(kernel_height), static_cast<size_t>(kernel_width)});
  auto rounding_type =
      ceil_mode ? ov::op::RoundingType::CEIL : ov::op::RoundingType::FLOOR;
  std::shared_ptr<Node> pool2d_op{nullptr};
  if (operation->type == NNADAPTER_AVERAGE_POOL_2D) {
    if (global_pooling) {
      auto axes_tensor = converter->AddConstantTensor(
          std::vector<int64_t>({input_operand->type.dimensions.count - 2,
                                input_operand->type.dimensions.count - 1}));
      pool2d_op = std::make_shared<default_opset::ReduceMean>(
          *input_tensor, *axes_tensor, true);
    } else {
      pool2d_op = std::make_shared<default_opset::AvgPool>(*input_tensor,
                                                           ov_strides,
                                                           ov_pads_begin,
                                                           ov_pads_end,
                                                           ov_kernel,
                                                           flag,
                                                           rounding_type,
                                                           ov_auto_pad);
    }
  } else if (operation->type == NNADAPTER_MAX_POOL_2D) {
    if (global_pooling) {
      auto axes_tensor = converter->AddConstantTensor(
          std::vector<int64_t>({input_operand->type.dimensions.count - 2,
                                input_operand->type.dimensions.count - 1}));
      pool2d_op = std::make_shared<default_opset::ReduceMax>(
          *input_tensor, *axes_tensor, true);
    } else {
      pool2d_op = std::make_shared<default_opset::MaxPool>(*input_tensor,
                                                           ov_strides,
                                                           ov::Strides({1, 1}),
                                                           ov_pads_begin,
                                                           ov_pads_end,
                                                           ov_kernel,
                                                           rounding_type,
                                                           ov_auto_pad);
    }
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported pooling operation type "
                         << OperationTypeToString(operation->type)
                         << " is found.";
  }
  MAP_OUTPUT(output_operand, pool2d_op, 0);
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
