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

#include <cmath>
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/nnadapter/bridges/converter.h"
#include "lite/kernels/nnadapter/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace nnadapter {

int ScaleConverter(void* ctx, OpLite* op, KernelBase* kernel) {
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
  float scale = op_info->GetAttr<float>("scale");
  float bias = op_info->GetAttr<float>("bias");
  bool bias_after_scale = op_info->GetAttr<bool>("bias_after_scale");
  if (!bias_after_scale) {
    bias *= scale;
  }
  auto has_scale = fabs(scale - 1.0f) > 1e-6f;
  auto has_bias = fabs(bias) > 1e-6f;

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
  if (!has_scale && !has_bias) {
    CHECK_LE(fabs(x_scale - out_scale), 1e-6f);
    converter->AddOperand(input_operand, out_name);
  } else {
    NNAdapterOperand* output_operand = nullptr;
    if (has_out_scale) {
      output_operand =
          converter->AddQuant8VariableOperand(out_dims, out_scale, out_name);
    } else {
      output_operand = converter->AddFloat32VariableOperand(out_dims, out_name);
    }
    if (has_scale) {
      // Scale operand
      NNAdapterOperand* scale_operand = nullptr;
      if (has_x_scale) {
        int8_t quant_scale_data = scale > 0.0f ? 1 : -1;
        scale_operand = converter->AddQuant8ConstantOperand(
            &quant_scale_data, DDim({static_cast<int64_t>(1)}), fabs(scale));
      } else {
        scale_operand = converter->AddFloat32ConstantOperand(
            &scale, DDim({static_cast<int64_t>(1)}));
      }
      // Fuse code operand
      auto fuse_code_operand =
          converter->AddInt32ConstantOperand(NNADAPTER_FUSED_NONE);
      // Immediate operand for input*scale
      NNAdapterOperand* immediate_operand = output_operand;
      if (has_bias) {
        if (has_x_scale) {
          immediate_operand = converter->AddQuant8VariableOperand(
              x_dims, x_scale * fabs(scale));
        } else {
          immediate_operand = converter->AddFloat32VariableOperand(x_dims);
        }
      }
      // Mul operation for input*scale
      std::vector<NNAdapterOperand*> input_operands = {
          input_operand, scale_operand, fuse_code_operand};
      std::vector<NNAdapterOperand*> output_operands = {immediate_operand};
      auto eltwise_mul_operation = converter->AddOperation(NNADAPTER_MUL);
      converter->SetOperation(
          eltwise_mul_operation, &input_operands, &output_operands);
      input_operand = immediate_operand;
    }
    if (has_bias) {
      // Bias operand
      NNAdapterOperand* bias_operand = nullptr;
      if (has_x_scale) {
        int8_t quant_bias_data = bias > 0.0f ? 1 : -1;
        bias_operand = converter->AddQuant8ConstantOperand(
            &quant_bias_data, DDim({static_cast<int64_t>(1)}), fabs(bias));
      } else {
        bias_operand = converter->AddFloat32ConstantOperand(
            &bias, DDim({static_cast<int64_t>(1)}));
      }
      // Fuse code operand
      auto fuse_code_operand =
          converter->AddInt32ConstantOperand(NNADAPTER_FUSED_NONE);
      // Add operation for input+bias or input*scale+bias
      std::vector<NNAdapterOperand*> input_operands = {
          input_operand, bias_operand, fuse_code_operand};
      std::vector<NNAdapterOperand*> output_operands = {output_operand};
      auto eltwise_add_operation = converter->AddOperation(NNADAPTER_ADD);
      converter->SetOperation(
          eltwise_add_operation, &input_operands, &output_operands);
    }
  }
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(scale,
                         kNNAdapter,
                         paddle::lite::subgraph::nnadapter::ScaleConverter);
