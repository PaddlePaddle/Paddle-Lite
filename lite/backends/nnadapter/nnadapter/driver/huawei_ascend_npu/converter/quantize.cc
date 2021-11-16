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

#include "core/operation/quantize.h"
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertQuantize(Converter* converter, hal::Operation* operation) {
  QUANTIZE_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK(is_per_layer_quant)
      << "HuaweiAscendNPU only support per layer quantize.";

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (input_operator == nullptr) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  auto quantize_op =
      converter->AddOperator<ge::op::AscendQuant>(output_operand);
  quantize_op->set_attr_scale(1.f / scale_data[0]);
  quantize_op->set_attr_offset(0.);
  SET_INPUT(quantize_op, x, input_operator);
  MAP_OUTPUT(quantize_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
