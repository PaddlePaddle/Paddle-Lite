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

int ConvertArgMinMax(Converter* converter, OpInfo* op, Scope* scope) {
  // Input operand
  auto x_name = op->Input("X").front();
  auto x_scale_name = "X0_scale";
  std::vector<float> x_scales;
  if (op->HasInputScale(x_scale_name, true)) {
    x_scales = op->GetInputScale(x_scale_name, true);
    CHECK(IsValidSymmPerLayerQuantParams(x_scales));
  }
  auto input_operand = converter->AddInputOperand(scope, x_name, {}, x_scales);

  // Axis operand
  int axis = op->GetAttr<int64_t>("axis");
  auto axis_operand = converter->AddConstantOperand(axis);

  // Keepdim operand
  bool keepdims = false;
  if (op->HasAttr("keepdims")) {
    keepdims = op->GetAttr<bool>("keepdims");
  }
  auto keepdim_operand =
      converter->AddConstantOperand(static_cast<int8_t>(keepdims));

  // Dtype operand, using int64 by default
  int dtype = 3;
  if (op->HasAttr("dtype")) {
    dtype = op->GetAttr<int>("dtype");
    if (dtype < 0) {
      dtype = 3;  // INT64 in FluidType
    }
  }
  auto dtype_operand = converter->AddConstantOperand(
      static_cast<int32_t>(ConvertFluidDataTypeToNNPrecisionCode(dtype)));

  // Output operand
  auto out_name = op->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  std::vector<float> out_scales;
  if (op->HasOutputScale(out_scale_name, true)) {
    out_scales = op->GetOutputScale(out_scale_name, true);
    CHECK(IsValidSymmPerLayerQuantParams(out_scales));
  }
  auto output_operand = converter->AddOutputOperand(out_name, out_scales);

  // ArgMin or ArgMax operation
  NNAdapterOperationType arg_operation_type;
  auto op_type = op->Type();
  if (op_type == "arg_min") {
    arg_operation_type = NNADAPTER_ARG_MIN;
  } else if (op_type == "arg_max") {
    arg_operation_type = NNADAPTER_ARG_MAX;
  } else {
    LOG(WARNING) << "Unsupported arg operation type: " << op_type;
    return UNSUPPORTED_FEATURE;
  }
  converter->AddOperation(
      arg_operation_type,
      {input_operand, axis_operand, keepdim_operand, dtype_operand},
      {output_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
