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

int ConvertPad(Converter* converter, OpInfo* op, Scope* scope) {
  // Extract op attributes
  // Input
  auto x_name = op->Input("X").front();
  auto x_scale_name = "X0_scale";
  std::vector<float> x_scales;
  if (op->HasInputScale(x_scale_name, true)) {
    x_scales = op->GetInputScale(x_scale_name, true);
  }
  // Mode
  auto mode = op->GetAttr<std::string>("mode");
  // Value, pads
  float value;
  std::vector<int> pads;
  if (op->Type() == "pad2d") {
    if (HasInput(op, scope, "Paddings")) {
      LOG(FATAL) << "Op type:" << op->Type()
                 << "doesn't support 'Paddings' input";
      return UNSUPPORTED_FEATURE;
    }
    value = op->GetAttr<float>("pad_value");
    pads = op->GetAttr<std::vector<int>>("paddings");
  } else if (op->Type() == "pad3d") {
    if (HasInput(op, scope, "Paddings")) {
      LOG(FATAL) << "Op type:" << op->Type()
                 << "doesn't support 'Paddings' input";
      return UNSUPPORTED_FEATURE;
    }
    value = op->GetAttr<float>("value");
    pads = op->GetAttr<std::vector<int>>("paddings");
  } else {
    LOG(FATAL) << "Unsupported op type: " << op->Type();
    return UNSUPPORTED_FEATURE;
  }
  // Data format
  auto data_format = op->GetAttr<std::string>("data_format");
  std::vector<int> paddings;
  if (data_format == "NCDHW") {
    CHECK_EQ(pads.size(), 6);
    paddings = {
        0, 0, 0, 0, pads[4], pads[5], pads[2], pads[3], pads[0], pads[1]};
  } else if (data_format == "NDHWC") {
    CHECK_EQ(pads.size(), 6);
    paddings = {
        0, 0, pads[4], pads[5], pads[2], pads[3], pads[0], pads[1], 0, 0};
  } else if (data_format == "NCHW") {
    CHECK_EQ(pads.size(), 4);
    paddings = {0, 0, 0, 0, pads[0], pads[1], pads[2], pads[3]};
  } else if (data_format == "NHWC") {
    CHECK_EQ(pads.size(), 4);
    paddings = {0, 0, pads[0], pads[1], pads[2], pads[3], 0, 0};
  } else {
    LOG(FATAL) << "Unsupported data format: " << data_format;
    return UNSUPPORTED_FEATURE;
  }
  // Output
  auto output_name = op->Output("Out").front();
  auto output_scale_name = "Out0_scale";
  std::vector<float> output_scales;
  if (op->HasOutputScale(output_scale_name, true)) {
    output_scales = op->GetOutputScale(output_scale_name, true);
  }

  // Convert to NNAdapter operands and operation
  // Input operand
  auto input_operand = converter->AddInputOperand(scope, x_name, {}, x_scales);
  // Pads operand
  auto pads_operand = converter->AddConstantOperand(paddings);
  // Mode operand
  int mode_code = ConvertPadModeToNNPadModeCode(mode);
  auto mode_operand = converter->AddConstantOperand(mode_code);
  // Value operand
  auto value_operand = converter->AddConstantOperand(value);
  // Output operand
  auto output_operand = converter->AddOutputOperand(output_name, output_scales);
  // Pad operation
  converter->AddOperation(
      NNADAPTER_PAD,
      {input_operand, pads_operand, mode_operand, value_operand},
      {output_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
