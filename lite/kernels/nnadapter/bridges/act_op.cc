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

int ActConverter(void* ctx, OpLite* op, KernelBase* kernel) {
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
  NNAdapterOperandType output_type;
  memset(&output_type, 0, sizeof(NNAdapterOperandType));
  output_type.precision = NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER;
  output_type.symm_per_layer_params.scale = out_scale;
  ConvertDimensions(
      out_dims, output_type.dimensions, &output_type.dimension_count);
  auto output_operand = converter->AddOperand(&output_type, out_name);

  // Activation operation
  std::vector<NNAdapterOperand*> input_operands = {input_operand};
  std::vector<NNAdapterOperand*> output_operands = {output_operand};
  NNAdapterOperation* activation_operation = nullptr;
  if (op_type == "sigmoid") {
    activation_operation = converter->AddOperation(NNADAPTER_SIGMOID);
  } else if (op_type == "relu") {
    activation_operation = converter->AddOperation(NNADAPTER_RELU);
  } else if (op_type == "relu6") {
    activation_operation = converter->AddOperation(NNADAPTER_RELU6);
  } else if (op_type == "tanh") {
    activation_operation = converter->AddOperation(NNADAPTER_TANH);
  } else {
    LOG(WARNING) << "Unsupported activation type: " << op_type;
    return FAILED;
  }
  converter->SetOperation(
      activation_operation, &input_operands, &output_operands);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(relu,
                         kNNAdapter,
                         paddle::lite::subgraph::nnadapter::ActConverter);
REGISTER_SUBGRAPH_BRIDGE(sigmoid,
                         kNNAdapter,
                         paddle::lite::subgraph::nnadapter::ActConverter);
REGISTER_SUBGRAPH_BRIDGE(relu6,
                         kNNAdapter,
                         paddle::lite::subgraph::nnadapter::ActConverter);
REGISTER_SUBGRAPH_BRIDGE(tanh,
                         kNNAdapter,
                         paddle::lite::subgraph::nnadapter::ActConverter);
