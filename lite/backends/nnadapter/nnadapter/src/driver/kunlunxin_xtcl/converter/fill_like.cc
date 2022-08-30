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

#include "operation/fill_like.h"
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertFillLike(Converter* converter, core::Operation* operation) {
  FILL_LIKE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to XTCL exprs
  auto input_expr = converter->GetMappedExpr(input_operand);
  if (!input_expr.defined()) {
    input_expr = converter->ConvertOperand(input_operand);
  }
  // The fill value should be a scalar constant
  xtcl::xExpr value_constant_expr;
  switch (value_operand->type.precision) {
    case NNADAPTER_BOOL8: {
      bool constant_value = *reinterpret_cast<int8_t*>(value_operand->buffer);
      NNADAPTER_VLOG(5) << "constant_value: " << constant_value;
      value_constant_expr =
          converter->builder()->CreateConstant({}, constant_value);
      break;
    }
    case NNADAPTER_INT8: {
      auto constant_value = *reinterpret_cast<int8_t*>(value_operand->buffer);
      NNADAPTER_VLOG(5) << "constant_value: " << constant_value;
      value_constant_expr =
          converter->builder()->CreateConstant({}, constant_value);
      break;
    }
    case NNADAPTER_INT32: {
      auto constant_value = *reinterpret_cast<int32_t*>(value_operand->buffer);
      NNADAPTER_VLOG(5) << "constant_value: " << constant_value;
      value_constant_expr =
          converter->builder()->CreateConstant({}, constant_value);
      break;
    }
    case NNADAPTER_INT64: {
      auto constant_value = *reinterpret_cast<int64_t*>(value_operand->buffer);
      NNADAPTER_VLOG(5) << "constant_value: " << constant_value;
      value_constant_expr =
          converter->builder()->CreateConstant({}, constant_value);
      break;
    }
    case NNADAPTER_FLOAT32: {
      auto constant_value = *reinterpret_cast<float*>(value_operand->buffer);
      NNADAPTER_VLOG(5) << "constant_value: " << constant_value;
      value_constant_expr =
          converter->builder()->CreateConstant({}, constant_value);
      break;
    }
    default: {
      NNADAPTER_LOG(FATAL) << "Unsupported precision: "
                           << value_operand->type.precision;
      break;
    }
  }
  // Paddle: data type of output is same as fill_value
  // XTCL:   data type of output is same as input_value
  if (input_operand->type.precision != value_operand->type.precision) {
    input_expr = converter->builder()->CreateCast(
        input_expr, ConvertToXTCLDataType(value_operand->type.precision));
  }
  auto fill_like_expr =
      converter->builder()->CreateFullLike(input_expr, value_constant_expr);
  converter->UpdateExprMap(output_operand, fill_like_expr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
