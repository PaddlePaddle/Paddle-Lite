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

#include "operation/fully_connected.h"
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/utility.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertFullyConnected(Converter* converter, core::Operation* operation) {
  FULLY_CONNECTED_OPERATION_EXTRACT_INPUTS_OUTPUTS
  auto batch_size =
      ProductionOfDimensions(input_operand->type.dimensions.data,
                             input_operand->type.dimensions.count) /
      input_size;
  NNADAPTER_VLOG(5) << "batch_size: " << batch_size;

  // Convert to XTCL exprs
  auto input_expr = converter->GetMappedExpr(input_operand);
  if (!input_expr.defined()) {
    input_expr = converter->ConvertOperand(input_operand);
  }
  // Reshape the input operator to 2-D tensor {batch_size, input_size} if the
  // dimensions_count not equal 2
  auto flatten_input_expr = input_expr;
  if (input_operand->type.dimensions.count != 2) {
    flatten_input_expr = converter->builder()->CreateReshape(
        input_expr, {-1, static_cast<int>(input_size)});
  }
  auto weight_expr = converter->ConvertOperand(weight_operand);
  auto bias_expr = converter->ConvertOperand(bias_operand);
  auto matmul2d_expr = converter->builder()->CreateMatmul2D(
      flatten_input_expr, weight_expr, true);
  auto bias_add_expr =
      matmul2d_expr;  // converter->builder()->CreateBiasAdd(matmul2d_expr, 1,
                      // bias_expr);
  converter->UpdateExprMap(output_operand, bias_add_expr);
  // fuse activations
  switch (fuse_code) {
#define CONVERT_UNARY_ACTIVATION(type, func)                              \
  case NNADAPTER_FUSED_##type:                                            \
    converter->UpdateExprMap(output_operand, converter->builder()->func); \
    break;
    CONVERT_UNARY_ACTIVATION(RELU, CreateRelu(bias_add_expr));
    CONVERT_UNARY_ACTIVATION(RELU6, CreateRelu6(bias_add_expr));
#undef CONVERT_UNARY_ACTIVATION
    case NNADAPTER_FUSED_NONE:
      break;
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported fuse_code(" << fuse_code
                           << ") is found.";
      break;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
