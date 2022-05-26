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

#include "operation/slice.h"
#include <limits.h>
#include "driver/intel_openvino/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
namespace nnadapter {
namespace intel_openvino {

int ConvertSlice(Converter* converter, core::Operation* operation) {
  SLICE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto start_idx_tensor = converter->AddConstantTensor(
      std::vector<int32_t>(starts, starts + starts_count));
  auto end_idx_tensor = converter->AddConstantTensor(
      std::vector<int32_t>(ends, ends + ends_count));
  auto strides_idx_tensor = converter->AddConstantTensor(
      std::vector<int32_t>(steps, steps + steps_count));

  // The following process is:
  // Given:
  // data = [ [1, 2, 3, 4], [5, 6, 7, 8], ] // shape is: [2, 4]
  // axes = [0]
  // starts = [1]
  // ends = [2]
  // Our process is:
  //  1. Get 'axes': [0, 1], 'starts', 'ends'
  //  2. Get data shape: [2,4] and dims: 2
  //  3. Create two tensor t1 and t2, shape is the dims from step2: 2. t1: [0,
  //  0], t2: [INT_MAX, INT_MAX]
  //  4. Use 'ScatterNDUpdate' to update some elements in t1, the updated
  //  indexes are coming from 'axes', the contents
  //  are coming from 'starts', t1: [1, 0]; apply the similar process to t2
  //  5. Call 'StrideSlice' with t1 and t2
  // Why using ScatterNDUpdate is that 'axes' may be discontinuous.

  // The shape of input, such as [2, 4].
  const auto shape_op = std::make_shared<default_opset::ShapeOf>(
      *input_tensor, GetElementType<int32_t>());
  // The input dim, such as [2].
  const auto rank_op = std::make_shared<default_opset::ShapeOf>(
      shape_op, GetElementType<int32_t>());
  const auto const_0_tensor = converter->AddConstantTensor<int32_t>(0);
  const auto const_max_tensor = converter->AddConstantTensor<int32_t>(INT_MAX);
  const auto const_1_tensor = converter->AddConstantTensor<int32_t>(1);
  // t1: [0, 0]
  const auto start_op =
      std::make_shared<default_opset::Broadcast>(*const_0_tensor, rank_op);
  // t2: [INT_MAX, INT_MAX]
  const auto end_op =
      std::make_shared<default_opset::Broadcast>(*const_max_tensor, rank_op);
  const auto strides_op =
      std::make_shared<default_opset::Broadcast>(*const_1_tensor, rank_op);
  const auto axes_tensor = converter->AddConstantTensor(
      std::vector<int32_t>(axes, axes + axes_count), {axes_count, 1});
  // Update t1.
  const auto fixed_start_op = std::make_shared<default_opset::ScatterNDUpdate>(
      start_op, *axes_tensor, *start_idx_tensor);
  // Update t2.
  const auto fixed_end_op = std::make_shared<default_opset::ScatterNDUpdate>(
      end_op, *axes_tensor, *end_idx_tensor);
  const auto fixed_strides_op =
      std::make_shared<default_opset::ScatterNDUpdate>(
          strides_op, *axes_tensor, *strides_idx_tensor);
  auto stride_slice_op =
      std::make_shared<default_opset::StridedSlice>(*input_tensor,
                                                    fixed_start_op,
                                                    fixed_end_op,
                                                    fixed_strides_op,
                                                    std::vector<int64_t>{0},
                                                    std::vector<int64_t>{0});
  MAP_OUTPUT(output_operand, stride_slice_op, 0);
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
