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

int ConvertInstanceNorm(Converter* converter, OpInfo* op, Scope* scope) {
  // Input operand
  auto x_name = op->Input("X").front();
  auto x_scale_name = "X0_scale";
  std::vector<float> x_scales;
  if (op->HasInputScale(x_scale_name, true)) {
    x_scales = op->GetInputScale(x_scale_name, true);
  }
  auto input_operand = converter->AddInputOperand(scope, x_name, {}, x_scales);
  CHECK(input_operand);
  auto input_type = converter->GetOperandType(input_operand);
  auto input_channel_size = input_type->dimensions.data[1];
  CHECK(input_channel_size != NNADAPTER_UNKNOWN);
  // Bias operand
  NNAdapterOperand* bias_operand = nullptr;
  if (HasInput(op, scope, "Bias")) {
    auto bias_name = op->Input("Bias").front();
    auto bias_tensor = scope->FindMutableTensor(bias_name);
    CHECK(bias_tensor->persistable());
    bias_operand = converter->AddConstantOperand(*bias_tensor);
  } else {
    bias_operand = converter->AddConstantOperand(
        std::vector<float>(input_channel_size, 0));
  }
  // Scale operand
  NNAdapterOperand* scale_operand = nullptr;
  if (HasInput(op, scope, "Scale")) {
    auto scale_name = op->Input("Scale").front();
    auto scale_tensor = scope->FindMutableTensor(scale_name);
    CHECK(scale_tensor->persistable());
    scale_operand = converter->AddConstantOperand(*scale_tensor);
  } else {
    scale_operand = converter->AddConstantOperand(
        std::vector<float>(input_channel_size, 1));
  }
  // Epsilon operand
  auto epsilon = op->GetAttr<float>("epsilon");
  auto epsilon_operand = converter->AddConstantOperand(epsilon);
  // Fuse code operand
  int32_t fuse_code_value = NNADAPTER_FUSED_NONE;
  std::string act_type = "";
  if (op->HasAttr("activation_type")) {
    act_type = op->GetAttr<std::string>("activation_type");
    if (act_type == "relu") {
      fuse_code_value = NNADAPTER_FUSED_RELU;
      act_type = "";
    } else if (act_type == "relu6") {
      fuse_code_value = NNADAPTER_FUSED_RELU6;
      act_type = "";
    }
  }
  auto fuse_code_operand = converter->AddConstantOperand(fuse_code_value);
  // Output operand
  auto out_name = op->Output("Y").front();
  auto output_operand = converter->AddOutputOperand(out_name);
  // InstanceNorm operand
  converter->AddOperation(NNADAPTER_INSTANCE_NORMALIZATION,
                          {input_operand,
                           scale_operand,
                           bias_operand,
                           epsilon_operand,
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
