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

#include "operation/top_k.h"
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertTopK(Converter* converter, core::Operation* operation) {
  TOP_K_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK_EQ(axis,
                     static_cast<int>(input_operand->type.dimensions.count) - 1)
      << "HuaweiAscendNPU only support last dimsion.";

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  auto k_operator = converter->GetMappedOperator(k_operand);
  if (!k_operator) {
    k_operator = converter->ConvertOperand(k_operand);
  }
  if (k_operand->type.precision == NNADAPTER_INT64) {
    auto cast_op = converter->AddOperator<ge::op::Cast>(output_operand);
    cast_op->set_attr_dst_type(ConvertToGEPrecision(NNADAPTER_INT32));
    SET_INPUT(cast_op, x, k_operator);
    k_operator = MAP_OUTPUT(cast_op, y, output_operand);
  }

  auto top_k_op = converter->AddOperator<ge::op::TopKV2>(output_operand);
  top_k_op->set_attr_dim(axis);
  top_k_op->set_attr_largest(largest);
  top_k_op->set_attr_sorted(sorted);
  SET_INPUT(top_k_op, x, input_operator);
  SET_INPUT(top_k_op, k, k_operator);
  auto output_operator = MAP_OUTPUT(top_k_op, values, output_operand);
  bool need_indices_cast = false;
  // ge::op::TopKV2 only support indices_type = DT_INT32, so cast to int64 if
  // necessary
  if (indices_operand->type.precision == NNADAPTER_INT64) {
    need_indices_cast = true;
    indices_operand->type.precision = NNADAPTER_INT32;
  }
  auto indices_operator = MAP_OUTPUT(top_k_op, indices, indices_operand);

  // trick to avoid AscendNPU error
  auto cast_output_op = converter->AddOperator<ge::op::Cast>(output_operand);
  cast_output_op->set_attr_dst_type(
      ConvertToGEPrecision(output_operand->type.precision));
  SET_INPUT(cast_output_op, x, output_operator);
  MAP_OUTPUT(cast_output_op, y, output_operand);

  indices_operand->type.precision = NNADAPTER_INT64;
  auto cast_indices_op = converter->AddOperator<ge::op::Cast>(indices_operand);
  if (need_indices_cast) {
    cast_indices_op->set_attr_dst_type(ConvertToGEPrecision(NNADAPTER_INT64));
  } else {
    cast_indices_op->set_attr_dst_type(ConvertToGEPrecision(NNADAPTER_INT32));
  }
  SET_INPUT(cast_indices_op, x, indices_operator);
  MAP_OUTPUT(cast_indices_op, y, indices_operand);

  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
