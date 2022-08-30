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

#include "operation/fill.h"
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertFill(Converter* converter, core::Operation* operation) {
  FILL_OPERATION_EXTRACT_INPUTS_OUTPUTS

  NNADAPTER_CHECK(IsConstantOperand(shape_operand));
  NNADAPTER_CHECK(IsConstantOperand(value_operand));

  // Convert to XTCL exprs
  // Fill value should be a scalar
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
  // Shape
  auto shape_count =
      shape_operand->length / static_cast<uint32_t>(sizeof(int32_t));
  auto shape_data = reinterpret_cast<int32_t*>(shape_operand->buffer);
  auto fill_expr = converter->builder()->CreateFull(
      value_constant_expr,
      ConvertToXTCLArray<xtcl::Integer>(shape_data, shape_count),
      ConvertToXTCLDataType(value_operand->type.precision));
  converter->UpdateExprMap(output_operand, fill_expr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
