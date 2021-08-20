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

  // Input
  auto x_name = op_info->Input("X").front();
  auto x_scale_name = "X0_scale";
  auto has_x_scale = op_info->HasInputScale(x_scale_name, true);
  auto x_scale =
      has_x_scale ? op_info->GetInputScale(x_scale_name, true)[0] : 0.f;
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  // Output
  auto out_name = op_info->Output("Y").front();
  auto out_scale_name = "Y0_scale";
  auto has_out_scale = op_info->HasOutputScale(out_scale_name, true);
  auto out_scale =
      has_out_scale ? op_info->GetOutputScale(out_scale_name, true)[0] : 0.f;
  auto out = scope->FindMutableTensor(out_name);
  auto out_dims = out->dims();
  // Scale
  auto scale_name = op_info->Input("Scale").front();
  auto scale = scope->FindMutableTensor(scale_name);
  auto scale_dims = scale->dims();
  // Bias
  auto bias_name = op_info->Input("Bias").front();
  auto bias = scope->FindMutableTensor(bias_name);
  auto bias_dims = bias->dims();
  // Mean
  auto mean_name = op_info->Input("Mean").front();
  auto mean = scope->FindMutableTensor(mean_name);
  auto mean_dims = mean->dims();
  // Variance
  auto variance_name = op_info->Input("Variance").front();
  auto variance = scope->FindMutableTensor(variance_name);
  auto variance_dims = variance->dims();
  // Epsilon
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
  float* scale_data = reinterpret_cast<float*>(scale->mutable_data<float>());
  NNAdapterOperand* scale_operand =
      converter->AddFloat32ConstantOperand(scale_data, scale_dims);
  // Bias operand
  float* bias_data = reinterpret_cast<float*>(bias->mutable_data<float>());
  NNAdapterOperand* bias_operand =
      converter->AddFloat32ConstantOperand(bias_data, bias_dims);
  // Mean operand
  float* mean_data = reinterpret_cast<float*>(mean->mutable_data<float>());
  NNAdapterOperand* mean_operand =
      converter->AddFloat32ConstantOperand(mean_data, mean_dims);
  // Variance operand
  float* variance_data =
      reinterpret_cast<float*>(variance->mutable_data<float>());
  NNAdapterOperand* variance_operand =
      converter->AddFloat32ConstantOperand(variance_data, variance_dims);
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
  converter->AddOperation(
      NNADAPTER_BATCH_NORMALIZATION, &input_operands, &output_operands);
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
