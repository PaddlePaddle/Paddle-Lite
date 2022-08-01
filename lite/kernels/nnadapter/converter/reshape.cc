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

int ConvertReshape(Converter* converter, OpInfo* op, Scope* scope) {
  // Extract the inputs, outputs and attributes
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
  // Shape operand
  // "ShapeTensor"(input) > "Shape"(input) > "shape"(attr)
  NNAdapterOperand* shape_operand = nullptr;
  if (HasInput(op, scope, "ShapeTensor")) {
    std::vector<NNAdapterOperand*> shapes_operands;
    for (auto shapes_tensor_name : op->Input("ShapeTensor")) {
      auto shapes_tensor_scale_name = shapes_tensor_name + "_scale";
      std::vector<float> shapes_tensor_scales;
      if (op->HasInputScale(shapes_tensor_scale_name, true)) {
        shapes_tensor_scales =
            op->GetInputScale(shapes_tensor_scale_name, true);
      }
      auto shapes_operand = converter->AddInputOperand(
          scope, shapes_tensor_name, {}, shapes_tensor_scales);
      shapes_operands.push_back(shapes_operand);
    }
    auto axis_operand = converter->AddConstantOperand(0);
    shapes_operands.push_back(axis_operand);
    shape_operand = converter->AddOutputOperand(out_name + "/concat");
    // Concat operation
    converter->AddOperation(NNADAPTER_CONCAT, shapes_operands, {shape_operand});
  } else if (HasInput(op, scope, "Shape")) {
    auto shape_name = op->Input("Shape").front();
    shape_operand = converter->AddInputOperand(scope, shape_name);
  } else {
    std::vector<int> shape = op->GetAttr<std::vector<int>>("shape");
    shape_operand = converter->AddConstantOperand(shape);
  }
  // Output operand
  auto output_operand = converter->AddOutputOperand(out_name, out_scales);
  // Reshape operation
  converter->AddOperation(
      NNADAPTER_RESHAPE, {input_operand, shape_operand}, {output_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
