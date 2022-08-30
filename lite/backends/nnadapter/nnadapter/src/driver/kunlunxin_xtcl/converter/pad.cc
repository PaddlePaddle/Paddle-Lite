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

#include "operation/pad.h"
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertPad(Converter* converter, core::Operation* operation) {
  PAD_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK_EQ(mode, NNADAPTER_PAD_MODE_CONSTANT)
      << "XTCL only support constant mode";
  if (mode == NNADAPTER_PAD_MODE_REPLICATE) {
    mode = NNADAPTER_PAD_MODE_EDGE;
  }
  std::string mode_str;
  switch (mode) {
    case NNADAPTER_PAD_MODE_CONSTANT:
      mode_str = "constant";
      break;
    case NNADAPTER_PAD_MODE_REFLECT:
      mode_str = "reflect";
      break;
    case NNADAPTER_PAD_MODE_EDGE:
      mode_str = "edge";
      break;
    default:
      // Don't support 'circular' mode. circular -> NNADAPTER_PAD_MODE_NONE
      NNADAPTER_LOG(FATAL) << "Unsupported mode: " << mode;
      break;
  }
  NNADAPTER_VLOG(5) << "mode: " << mode_str;

  // Convert to XTCL exprs
  auto input_expr = converter->GetMappedExpr(input_operand);
  if (!input_expr.defined()) {
    input_expr = converter->ConvertOperand(input_operand);
  }
  // Must be constant expr
  NNADAPTER_CHECK_EQ(value_operand->type.dimensions.count, 1)
      << "Expect value_operand dimensions count: 1"
      << ", but receive: " << value_operand->type.dimensions.count;
  NNADAPTER_CHECK_EQ(value_operand->type.dimensions.data[0], 1)
      << "Expect value_operand shape: [1]"
      << ", but receive: " << value_operand->type.dimensions.data[0];
  xtcl::xExpr pade_value_constant_expr;
  // Value operand has the same type as 'input'
  // XTCL get value from attrs pad_val, not from input pad_val.
  // attrs pad_val must be scalar constant
  switch (input_operand->type.precision) {
    case NNADAPTER_INT32: {
      auto constant_value = *reinterpret_cast<int32_t*>(value_operand->buffer);
      NNADAPTER_VLOG(5) << "constant_value: " << constant_value;
      pade_value_constant_expr =
          converter->builder()->CreateConstant({}, constant_value);
      break;
    }
    case NNADAPTER_INT64: {
      auto constant_value = *reinterpret_cast<int64_t*>(value_operand->buffer);
      NNADAPTER_VLOG(5) << "constant_value: " << constant_value;
      pade_value_constant_expr = converter->builder()->CreateConstant(
          {}, static_cast<int>(constant_value));
      break;
    }
    case NNADAPTER_FLOAT32: {
      auto constant_value = *reinterpret_cast<float*>(value_operand->buffer);
      NNADAPTER_VLOG(5) << "constant_value: " << constant_value;
      pade_value_constant_expr =
          converter->builder()->CreateConstant({}, constant_value);
      break;
    }
    default: {
      NNADAPTER_LOG(FATAL) << "Unsupported precision: "
                           << input_operand->type.precision;
      break;
    }
  }
  // Tensor of shape [2 * rank(`input`)]
  uint32_t pads_size =
      pads_operand->length / static_cast<uint32_t>(sizeof(int32_t));
  NNADAPTER_CHECK_EQ((input_operand->type.dimensions.count) * 2, pads_size)
      << "Expect pads_size == 2 * rank(input)";
  xtcl::Array<xtcl::Array<xtcl::Integer>> pad_width;
  auto pads_buffer = reinterpret_cast<int32_t*>(pads_operand->buffer);
  for (size_t i = 0; i < static_cast<size_t>(pads_size); i = i + 2) {
    std::vector<int32_t> padding_vec;
    padding_vec.resize(2);
    padding_vec[0] = pads_buffer[i];
    padding_vec[1] = pads_buffer[i + 1];
    pad_width.push_back(ConvertToXTCLArray<xtcl::Integer>(padding_vec));
  }
  auto pad_expr = converter->builder()->CreatePad(
      input_expr, pad_width, pade_value_constant_expr, mode_str);
  converter->UpdateExprMap(output_operand, pad_expr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
