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

#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/nnadapter/bridges/converter.h"
#include "lite/kernels/nnadapter/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace nnadapter {

int HardSwishConverter(void* ctx, OpLite* op, KernelBase* kernel) {
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
  auto out_name = op_info->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  auto has_out_scale = op_info->HasOutputScale(out_scale_name, true);
  auto out_scale =
      has_out_scale ? op_info->GetOutputScale(out_scale_name, true)[0] : 0.f;
  auto out = scope->FindMutableTensor(out_name);
  auto out_dims = out->dims();
  float offset = op_info->GetAttr<float>("offset");
  float threshold = op_info->GetAttr<float>("threshold");
  float scale = op_info->GetAttr<float>("scale");
  float mul_factor = threshold / scale;
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
  // Output operand
  NNAdapterOperand* output_operand = nullptr;
  if (has_out_scale) {
    output_operand =
        converter->AddQuant8VariableOperand(out_dims, out_scale, out_name);
  } else {
    output_operand = converter->AddFloat32VariableOperand(out_dims, out_name);
  }
  // Activation operation
  std::vector<NNAdapterOperand*> input_operands = {input_operand};
  std::vector<NNAdapterOperand*> output_operands = {output_operand};
  NNAdapterOperation* hardswish_operation = nullptr;
  // Prepare data
  auto offset_operand = converter->AddFloat32ConstantOperand(offset);
  auto threshold_operand = converter->AddFloat32ConstantOperand(threshold);
  input_operands.push_back(offset_operand);
  input_operands.push_back(threshold_operand);
  // If mul_factor not equal to 1.0, split paddle operator to hard_swish and mul
  if (mul_factor == 1.0) {
    // Hard_swish operator
    hardswish_operation = converter->AddOperation(NNADAPTER_HARD_SWISH);
    converter->SetOperation(
        hardswish_operation, &input_operands, &output_operands);
  } else {
    // Immediate operand
    auto immediate_operand = converter->AddFloat32VariableOperand(x_dims);
    std::vector<NNAdapterOperand*> immediate_output_operands = {
        immediate_operand};
    // Hard_swish operator
    hardswish_operation = converter->AddOperation(NNADAPTER_HARD_SWISH);
    converter->SetOperation(
        hardswish_operation, &input_operands, &immediate_output_operands);
    // Mul operand
    auto mul_operand = converter->AddFloat32ConstantOperand(
        &mul_factor, DDim({static_cast<int64_t>(1)}));
    // Fuse code operand
    auto fuse_code_operand =
        converter->AddInt32ConstantOperand(NNADAPTER_FUSED_NONE);
    std::vector<NNAdapterOperand*> mul_input_operands = {
        immediate_operand, mul_operand, fuse_code_operand};
    // Elementwise mul operator
    auto elementwise_mul = converter->AddOperation(NNADAPTER_MUL);
    converter->SetOperation(
        elementwise_mul, &mul_input_operands, &output_operands);
  }

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(hard_swish,
                         kNNAdapter,
                         paddle::lite::subgraph::nnadapter::HardSwishConverter);
