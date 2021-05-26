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
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto out_name = op_info->Output("Out").front();
  auto out_scale_name = "Out0_scale";
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
  CHECK(op_info->HasInputScale(x_scale_name, true));
  auto x_scale = op_info->GetInputScale(x_scale_name, true)[0];
  NNAdapterOperand* input_operand = nullptr;
  if (converter->HasOperand(x_name)) {
    input_operand = converter->GetOperand(x_name);
  } else {
    NNAdapterOperandType input_type;
    memset(&input_type, 0, sizeof(NNAdapterOperandType));
    input_type.precision = NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER;
    input_type.symm_per_layer_params.scale = x_scale;
    ConvertDimensions(
        x_dims, input_type.dimensions, &input_type.dimension_count);
    input_operand = converter->AddOperand(&input_type, x_name);
  }

  // Output operand
  CHECK(op_info->HasOutputScale(out_scale_name, true));
  auto out_scale = op_info->GetOutputScale(out_scale_name, true)[0];
  if (!has_scale && !has_bias) {
    CHECK_LE(fabs(x_scale - out_scale), 1e-6f);
    converter->AddOperand(input_operand, out_name);
  } else {
    NNAdapterOperandType output_type;
    memset(&output_type, 0, sizeof(NNAdapterOperandType));
    output_type.precision = NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER;
    output_type.symm_per_layer_params.scale = out_scale;
    ConvertDimensions(
        out_dims, output_type.dimensions, &output_type.dimension_count);
    auto output_operand = converter->AddOperand(&output_type, out_name);
    if (has_scale) {
      // Scale operand
      NNAdapterOperandType scale_type;
      memset(&scale_type, 0, sizeof(NNAdapterOperandType));
      scale_type.precision = NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER;
      scale_type.symm_per_layer_params.scale = fabs(scale);
      scale_type.dimension_count = 1;
      scale_type.dimensions[0] = 1;
      auto scale_operand = converter->AddOperand(&scale_type);
      int8_t quant_scale_data = scale > 0.0f ? 1 : -1;
      converter->SetOperandCopyFrom(
          scale_operand, &quant_scale_data, sizeof(int8_t));

      // Immediate operand for input*scale
      NNAdapterOperand* immediate_operand = output_operand;
      if (has_bias) {
        NNAdapterOperandType immediate_type;
        memset(&immediate_type, 0, sizeof(NNAdapterOperandType));
        immediate_type.precision = NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER;
        immediate_type.symm_per_layer_params.scale = x_scale * fabs(scale);
        ConvertDimensions(
            x_dims, immediate_type.dimensions, &immediate_type.dimension_count);
        immediate_operand = converter->AddOperand(&immediate_type);
      }

      // Mul operation for input*scale
      std::vector<NNAdapterOperand*> input_operands = {input_operand,
                                                       scale_operand};
      std::vector<NNAdapterOperand*> output_operands = {immediate_operand};
      auto eltwise_mul_operation = converter->AddOperation(NNADAPTER_MUL);
      converter->SetOperation(
          eltwise_mul_operation, &input_operands, &output_operands);
      input_operand = immediate_operand;
    }
    if (has_bias) {
      // Bias operand
      NNAdapterOperandType bias_type;
      memset(&bias_type, 0, sizeof(NNAdapterOperandType));
      bias_type.precision = NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER;
      bias_type.symm_per_layer_params.scale = fabs(bias);
      bias_type.dimension_count = 1;
      bias_type.dimensions[0] = 1;
      auto bias_operand = converter->AddOperand(&bias_type);
      int8_t quant_bias_data = bias > 0.0f ? 1 : -1;
      converter->SetOperandCopyFrom(
          bias_operand, &quant_bias_data, sizeof(int8_t));

      // Add operation for input+bias or input*scale+bias
      std::vector<NNAdapterOperand*> input_operands = {input_operand,
                                                       bias_operand};
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
