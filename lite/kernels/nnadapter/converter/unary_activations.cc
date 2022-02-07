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

#include <cmath>
#include "lite/kernels/nnadapter/converter/converter.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace nnadapter {

int ConvertUnaryActivations(Converter* converter, OpInfo* op, Scope* scope) {
  // Extract the inputs, outputs and attributes
  auto op_type = op->Type();
  auto x_name = op->Input("X").front();
  auto x_scale_name = "X0_scale";
  std::vector<float> x_scales;
  if (op->HasInputScale(x_scale_name, true)) {
    x_scales = op->GetInputScale(x_scale_name, true);
  }
  auto out_name = op->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  std::vector<float> out_scales;
  if (op->HasOutputScale(out_scale_name, true)) {
    out_scales = op->GetOutputScale(out_scale_name, true);
  }
  // Check quantization mode
  bool is_quant_mode = IsValidSymmPerLayerQuantParams(out_scales);
  if (is_quant_mode) {
    CHECK(IsValidSymmPerLayerQuantParams(x_scales))
        << "Missing the quant params '" << x_scale_name << "' for the input '"
        << x_name << "'";
  }

  // Convert to NNAdapter operands and operation
  // Input operand
  auto input_operand = converter->AddInputOperand(scope, x_name, {}, x_scales);
  CHECK(input_operand);
  auto input_type = converter->GetOperandType(input_operand);
  if (is_quant_mode) {
    CHECK(IsNNInt8SymmPerLayerQuantType(*input_type));
    std::vector<float> quant_scales;
    CHECK(GetNNSymmQuantParams(*input_type, &quant_scales));
    CHECK(IsSameSymmQuantParams(x_scales, quant_scales));
  }
  // Output operand
  auto output_operand = converter->AddOutputOperand(out_name, out_scales);
  // Unary activation operations
  NNAdapterOperationType unary_act_operation_type;
  if (op_type == "swish") {
    auto beta = op->GetAttr<float>("beta");
    CHECK_LT(fabs(beta - 1.0f), 1e-5f) << "Only supports beta = 1.0";
    unary_act_operation_type = NNADAPTER_SWISH;
  } else {
    unary_act_operation_type = ConvertUnaryActTypeToNNOperationType(op_type);
    CHECK(unary_act_operation_type != NNADAPTER_UNKNOWN)
        << "Unsupported unary activation type: " << op_type;
  }
  converter->AddOperation(
      unary_act_operation_type, {input_operand}, {output_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
