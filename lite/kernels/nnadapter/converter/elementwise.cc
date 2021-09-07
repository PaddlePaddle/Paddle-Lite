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

#include "lite/kernels/nnadapter/converter/converter.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace nnadapter {

int ConvertElementwise(Converter* converter, OpInfo* op, Scope* scope) {
  // X operand
  auto x_name = op->Input("X").front();
  auto x_operand = converter->GetMappedOperand(x_name);

  // Y operand
  auto y_name = op->Input("Y").front();
  auto y_operand = converter->GetMappedOperand(y_name);

  // Output
  auto out_name = op->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  std::vector<float> out_scales;
  if (op->HasOutputScale(out_scale_name, true)) {
    out_scales = op->GetOutputScale(out_scale_name, true);
  }
  auto output_operand = converter->AddOutputOperand(out_name, out_scales);

  // Elementwise operation
  NNAdapterOperationType eltwise_operation_type;
  auto op_type = op->Type();
  if (op_type == "elementwise_add" ||
      op_type == "fusion_elementwise_add_activation") {
    eltwise_operation_type = NNADAPTER_ADD;
  } else if (op_type == "elementwise_sub" ||
             op_type == "fusion_elementwise_sub_activation") {
    eltwise_operation_type = NNADAPTER_SUB;
  } else if (op_type == "elementwise_mul" ||
             op_type == "fusion_elementwise_mul_activation") {
    eltwise_operation_type = NNADAPTER_MUL;
  } else if (op_type == "elementwise_div" ||
             op_type == "fusion_elementwise_div_activation") {
    eltwise_operation_type = NNADAPTER_DIV;
  } else if (op_type == "elementwise_max" ||
             op_type == "fusion_elementwise_max_activation") {
    eltwise_operation_type = NNADAPTER_MAX;
  } else if (op_type == "elementwise_min" ||
             op_type == "fusion_elementwise_min_activation") {
    eltwise_operation_type = NNADAPTER_MIN;
  } else if (op_type == "elementwise_pow" ||
             op_type == "fusion_elementwise_pow_activation") {
    eltwise_operation_type = NNADAPTER_POW;
  } else {
    LOG(WARNING) << "Unsupported elementwise op type: " << op_type;
    return UNSUPPORTED_FEATURE;
  }
  converter->AddOperation(
      eltwise_operation_type, {x_operand, y_operand}, {output_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
