// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/nnadapter/bridges/converter.h"
#include "lite/kernels/nnadapter/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace nnadapter {

int BatchNormalizationConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto converter = static_cast<Converter*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "Converting " << op_type << " ...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x_scale_name = "X0_scale";
  auto has_x_scale = op_info->HasInputScale(x_scale_name, true);
  auto x_scale =
      has_x_scale ? op_info->GetInputScale(x_scale_name, true)[0] : 0.f;
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();

  auto out_name = op_info->Output("Y").front();
  auto out_scale_name = "Y0_scale";
  auto has_out_scale = op_info->HasOutputScale(out_scale_name, true);
  auto out_scale =
      has_out_scale ? op_info->GetOutputScale(out_scale_name, true)[0] : 0.f;
  auto out = scope->FindMutableTensor(out_name);
  auto out_dims = out->dims();

  auto scale_name = op_info->Input("Scale").front();
  auto scale = scope->FindMutableTensor(scale_name);
  auto scale_dims = scale->dims();
  auto bias_name = op_info->Input("Bias").front();
  auto bias = scope->FindMutableTensor(bias_name);
  auto bias_dims = bias->dims();
  auto mean_name = op_info->Input("Mean").front();
  auto mean = scope->FindMutableTensor(mean_name);
  auto mean_dims = mean->dims();
  auto variance_name = op_info->Input("Variance").front();
  auto variance = scope->FindMutableTensor(variance_name);
  auto variance_dims = variance->dims();
  float epsilon = op_info->GetAttr<float>("epsilon");

  // Input operand
  NNAdapterOperand* input_operand = nullptr;
  if (converter->HasOperand(x_name)) {
    input_operand = converter->GetOperand(x_name);
  } else {
    if (has_x_scale) {
      input_operand =
          converter->AddQuant8VariableOperand(x_dims, x_scale, x_name);
    } else {
      input_operand = converter->AddFloat32VariableOperand(x_dims, x_name);
    }
  }

  // Scale operand
  NNAdapterOperand* scale_operand = nullptr;
  if (converter->HasOperand(scale_name)) {
    scale_operand = converter->GetOperand(scale_name);
  } else {
    scale_operand =
        converter->AddFloat32VariableOperand(scale_dims, scale_name);
  }

  // Bias operand
  NNAdapterOperand* bias_operand = nullptr;
  if (converter->HasOperand(bias_name)) {
    bias_operand = converter->GetOperand(bias_name);
  } else {
    bias_operand = converter->AddFloat32VariableOperand(bias_dims, bias_name);
  }

  // Mean operand
  NNAdapterOperand* mean_operand = nullptr;
  if (converter->HasOperand(mean_name)) {
    mean_operand = converter->GetOperand(mean_name);
  } else {
    mean_operand = converter->AddFloat32VariableOperand(mean_dims, mean_name);
  }

  // Variance operand
  NNAdapterOperand* variance_operand = nullptr;
  if (converter->HasOperand(variance_name)) {
    variance_operand = converter->GetOperand(variance_name);
  } else {
    variance_operand =
        converter->AddFloat32VariableOperand(variance_dims, variance_name);
  }

  // epsilon operand
  auto epsilon_operand = converter->AddFloat32ConstantOperand(epsilon);

  // Output operand
  NNAdapterOperand* output_operand = nullptr;
  if (has_out_scale) {
    output_operand =
        converter->AddQuant8VariableOperand(out_dims, out_scale, out_name);
  } else {
    output_operand = converter->AddFloat32VariableOperand(out_dims, out_name);
  }

  // BatchNorm operation
  std::vector<NNAdapterOperand*> input_operands = {input_operand,
                                                   scale_operand,
                                                   bias_operand,
                                                   mean_operand,
                                                   variance_operand,
                                                   epsilon_operand};
  std::vector<NNAdapterOperand*> output_operands = {output_operand};
  NNAdapterOperation* batch_norm_operation =
      converter->AddOperation(NNADAPTER_BATCH_NORMALIZATION);

  converter->SetOperation(
      batch_norm_operation, &input_operands, &output_operands);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    batch_norm,
    kNNAdapter,
    paddle::lite::subgraph::nnadapter::BatchNormalizationConverter);
