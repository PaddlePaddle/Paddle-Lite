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

int SoftmaxConverter(void* ctx, OpLite* op, KernelBase* kernel) {
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
  auto x_rank = x_dims.size();
  CHECK_GE(x_rank, 2UL);
  auto out_name = op_info->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  auto out = scope->FindMutableTensor(out_name);
  auto out_dims = out->dims();
  auto axis = op_info->GetAttr<int>("axis");
  if (axis < 0) {
    axis += x_rank;
  }

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

  // Axis operand
  NNAdapterOperandType int32_type;
  memset(&int32_type, 0, sizeof(NNAdapterOperandType));
  int32_type.precision = NNADAPTER_INT32;
  int32_type.dimension_count = 0;
  auto axis_operand = converter->AddOperand(&int32_type);
  converter->SetOperandCopyFrom(axis_operand, &axis, sizeof(int32_t));

  // Output operand
  CHECK(op_info->HasOutputScale(out_scale_name, true));
  auto out_scale = op_info->GetOutputScale(out_scale_name, true)[0];
  NNAdapterOperandType output_type;
  memset(&output_type, 0, sizeof(NNAdapterOperandType));
  output_type.precision = NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER;
  output_type.symm_per_layer_params.scale = out_scale;
  ConvertDimensions(
      out_dims, output_type.dimensions, &output_type.dimension_count);
  auto output_operand = converter->AddOperand(&output_type, out_name);

  // Softmax operation
  std::vector<NNAdapterOperand*> input_operands = {input_operand, axis_operand};
  std::vector<NNAdapterOperand*> output_operands = {output_operand};
  auto softmax_operation = converter->AddOperation(NNADAPTER_SOFTMAX);
  converter->SetOperation(softmax_operation, &input_operands, &output_operands);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(softmax,
                         kNNAdapter,
                         paddle::lite::subgraph::nnadapter::SoftmaxConverter);
