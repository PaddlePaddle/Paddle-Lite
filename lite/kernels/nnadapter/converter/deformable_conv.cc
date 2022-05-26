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

int ConvertDeformableConv(Converter* converter, OpInfo* op, Scope* scope) {
  // Input operand
  auto input_name = op->Input("Input").front();
  auto input_scale_name = "Input0_scale";
  std::vector<float> input_scales;
  if (op->HasInputScale(input_scale_name, true)) {
    input_scales = op->GetInputScale(input_scale_name, true);
  }
  auto input_operand =
      converter->AddInputOperand(scope, input_name, {}, input_scales);

  // Offset operand
  auto offset_name = op->Input("Offset").front();
  auto offset_operand = converter->AddInputOperand(scope, offset_name);

  // Mask operand
  auto mask_name = op->Input("Mask").front();
  auto mask_operand = converter->AddInputOperand(scope, mask_name);

  // Filter operand
  auto filter_name = op->Input("Filter").front();
  auto filter_tensor = scope->FindTensor(filter_name);
  auto filter_operand = converter->AddConstantOperand(*filter_tensor);

  // Bias operand
  NNAdapterOperand* bias_operand = nullptr;
  if (HasInput(op, scope, "Bias")) {
    auto bias_name = op->Input("Bias").front();
    auto bias_tensor = scope->FindTensor(bias_name);
    bias_operand = converter->AddConstantOperand(*bias_tensor);
  } else {
    // Add dummy zero bias operand
    std::vector<float> zeros(filter_tensor->dims()[0], 0.f);
    bias_operand = converter->AddConstantOperand(zeros);
  }

  // Pads operand
  std::vector<int> pads = op->GetAttr<std::vector<int>>("paddings");
  if (pads.size() == 2UL) {
    pads.insert(pads.begin() + 1, pads[0]);
    pads.insert(pads.begin() + 3, pads[2]);
  }
  CHECK_EQ(pads.size(), 4UL);
  auto pads_operand = converter->AddConstantOperand(pads);

  // Strides operand
  std::vector<int> strides = op->GetAttr<std::vector<int>>("strides");
  CHECK_EQ(strides.size(), 2UL);
  auto strides_operand = converter->AddConstantOperand(strides);

  // Group operand
  int group = op->GetAttr<int>("groups");
  auto group_operand = converter->AddConstantOperand(group);

  // Deformable_group operand
  int deformable_group = op->GetAttr<int>("deformable_groups");
  auto deformable_group_operand =
      converter->AddConstantOperand(deformable_group);

  // Dilations operand
  std::vector<int> dilations = op->GetAttr<std::vector<int>>("dilations");
  CHECK_EQ(dilations.size(), 2UL);
  auto dilations_operand = converter->AddConstantOperand(dilations);

  // Fuse code operand
  bool with_act = op->HasAttr("with_act") && op->GetAttr<bool>("with_act");
  std::string act_type = with_act ? op->GetAttr<std::string>("act_type") : "";
  int32_t fuse_code_value = NNADAPTER_FUSED_NONE;
  if (act_type == "relu") {
    fuse_code_value = NNADAPTER_FUSED_RELU;
    act_type = "";
  } else if (act_type == "relu1") {
    fuse_code_value = NNADAPTER_FUSED_RELU1;
    act_type = "";
  } else if (act_type == "relu6") {
    fuse_code_value = NNADAPTER_FUSED_RELU6;
    act_type = "";
  }
  auto fuse_code_operand = converter->AddConstantOperand(fuse_code_value);

  // Output operand
  auto out_name = op->Output("Output").front();
  auto output_operand = converter->AddOutputOperand(out_name);

  // Deformable_conv2d operation
  converter->AddOperation(NNADAPTER_DEFORMABLE_CONV_2D,
                          {input_operand,
                           offset_operand,
                           mask_operand,
                           filter_operand,
                           bias_operand,
                           pads_operand,
                           strides_operand,
                           group_operand,
                           deformable_group_operand,
                           dilations_operand,
                           fuse_code_operand},
                          {output_operand});

  // Unpack the fused activations
  if (!act_type.empty()) {
    auto fused_act_output_operand = converter->AddOutputOperand(out_name);
    if (act_type == "leaky_relu") {
      auto alpha = op->GetAttr<float>("leaky_relu_alpha");
      auto alpha_operand = converter->AddConstantOperand(alpha);
      converter->AddOperation(NNADAPTER_LEAKY_RELU,
                              {output_operand, alpha_operand},
                              {fused_act_output_operand});
    } else {
      // Unpack the fused unary activations
      auto unary_act_operation_type =
          ConvertUnaryActTypeToNNOperationType(act_type);
      CHECK(unary_act_operation_type != NNADAPTER_UNKNOWN)
          << "Failed to unpack the fused activation type: " << act_type;
      converter->AddOperation(unary_act_operation_type,
                              {output_operand},
                              {fused_act_output_operand});
    }
  }
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
