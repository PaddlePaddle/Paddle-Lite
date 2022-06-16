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
// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "operation/flatten.h"
#include "driver/intel_openvino/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace intel_openvino {

int ConvertFlatten(Converter* converter, core::Operation* operation) {
  FLATTEN_OPERATION_EXTRACT_INPUTS_OUTPUTS

  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  if (start_axis < 0) start_axis += input_operand->type.dimensions.count;
  if (end_axis < 0) end_axis += input_operand->type.dimensions.count;
  auto shape_of_x = std::make_shared<default_opset::ShapeOf>(*input_tensor);
  int dims = input_tensor->get_partial_shape().rank().get_length();
  auto axis1_begin_tensor =
      converter->AddConstantTensor<int64_t>(std::vector<int64_t>{0});
  auto axis1_end_tensor =
      converter->AddConstantTensor<int64_t>(std::vector<int64_t>{start_axis});
  auto axis1 =
      std::make_shared<default_opset::StridedSlice>(shape_of_x,
                                                    *axis1_begin_tensor,
                                                    *axis1_end_tensor,
                                                    std::vector<int64_t>{0},
                                                    std::vector<int64_t>{0});
  TensorVector axes{
      axis1, *converter->AddConstantTensor<int64_t>(std::vector<int64_t>{-1})};
  if (end_axis < dims - 1) {
    auto axis2_begin_tensor = converter->AddConstantTensor<int64_t>(
        std::vector<int64_t>{end_axis + 1});
    auto axis2_end_tensor =
        converter->AddConstantTensor<int64_t>(std::vector<int64_t>{dims});
    auto axis2 =
        std::make_shared<default_opset::StridedSlice>(shape_of_x,
                                                      *axis2_begin_tensor,
                                                      *axis2_end_tensor,
                                                      std::vector<int64_t>{0},
                                                      std::vector<int64_t>{0});
    axes.push_back(axis2);
  }
  auto new_shape_op = std::make_shared<default_opset::Concat>(axes, 0);
  auto reshape_op = std::make_shared<default_opset::Reshape>(
      *input_tensor, new_shape_op->output(0), false);
  MAP_OUTPUT(output_operand, reshape_op, 0);
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
