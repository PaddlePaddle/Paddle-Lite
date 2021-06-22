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

int ElementwiseConverter(void* ctx, OpLite* op, KernelBase* kernel) {
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
  auto y_name = op_info->Input("Y").front();
  auto y_scale_name = "Y0_scale";
  auto has_y_scale = op_info->HasInputScale(y_scale_name, true);
  auto y_scale =
      has_y_scale ? op_info->GetInputScale(y_scale_name, true)[0] : 0.f;
  auto y = scope->FindMutableTensor(y_name);
  auto y_dims = y->dims();
  auto out_name = op_info->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  auto has_out_scale = op_info->HasOutputScale(out_scale_name, true);
  auto out_scale =
      has_out_scale ? op_info->GetOutputScale(out_scale_name, true)[0] : 0.f;
  auto out = scope->FindMutableTensor(out_name);
  auto out_dims = out->dims();
  auto axis = op_info->GetAttr<int>("axis");
  if (axis < 0) {
    axis = x_dims.size() - y_dims.size();
  }
  auto act_type = op_info->HasAttr("act_type")
                      ? op_info->GetAttr<std::string>("act_type")
                      : "";
  // Check whether the two dimensions are compatiable(Numpy-style broadcasting
  // https://numpy.org/doc/stable/user/basics.broadcasting.html).
  // Fill the dimension of Y with 1
  std::vector<int64_t> y_shape(x_dims.size(), 1);
  for (int i = axis; i < x_dims.size(); i++) {
    if (i < axis + y_dims.size()) {
      if (x_dims[i] != y_dims[i - axis] && x_dims[i] != 1 &&
          y_dims[i - axis] != 1) {
        LOG(ERROR) << "Incompatible broadcasting at " << i << " with axis "
                   << axis << ", expect " << x_dims[i] << " but received "
                   << y_dims[i - axis] << ".";
        return FAILED;
      } else {
        y_shape[i] = x_dims[i];
      }
    }
  }

  // Input0 operand
  NNAdapterOperand* input0_operand = nullptr;
  if (converter->HasOperand(x_name)) {
    input0_operand = converter->GetOperand(x_name);
  } else {
    if (has_x_scale) {
      input0_operand =
          converter->AddQuant8VariableOperand(x_dims, x_scale, x_name);
    } else {
      input0_operand = converter->AddFloat32VariableOperand(x_dims, x_name);
    }
  }

  // Input1 operand
  NNAdapterOperand* input1_operand = nullptr;
  if (converter->HasOperand(y_name)) {
    input1_operand = converter->GetOperand(y_name);
  } else {
    if (has_y_scale) {
      input1_operand =
          converter->AddQuant8VariableOperand(DDim(y_shape), y_scale, y_name);
    } else {
      input1_operand =
          converter->AddFloat32VariableOperand(DDim(y_shape), y_name);
    }
  }

  // Fuse code operand
  int32_t fuse_code_value = NNADAPTER_FUSED_NONE;
  if (act_type == "relu") {
    fuse_code_value = 1;
  } else if (act_type == "relu1") {
    fuse_code_value = 2;
  } else if (act_type == "relu6") {
    fuse_code_value = 3;
  } else if (!act_type.empty()) {
    LOG(WARNING) << "Unsupported activation type: " << act_type;
    return FAILED;
  }
  auto fuse_code_operand = converter->AddInt32ConstantOperand(fuse_code_value);

  // Output operand
  NNAdapterOperand* output_operand = nullptr;
  if (has_out_scale) {
    output_operand =
        converter->AddQuant8VariableOperand(out_dims, out_scale, out_name);
  } else {
    output_operand = converter->AddFloat32VariableOperand(out_dims, out_name);
  }

  // ADD, SUB, MUL and DIV operation
  std::vector<NNAdapterOperand*> input_operands = {
      input0_operand, input1_operand, fuse_code_operand};
  std::vector<NNAdapterOperand*> output_operands = {output_operand};
  NNAdapterOperation* elementwise_operation = nullptr;
  if (op_type == "elementwise_add" ||
      op_type == "fusion_elementwise_add_activation") {
    elementwise_operation = converter->AddOperation(NNADAPTER_ADD);
  } else if (op_type == "elementwise_sub" ||
             op_type == "fusion_elementwise_sub_activation") {
    elementwise_operation = converter->AddOperation(NNADAPTER_SUB);
  } else if (op_type == "elementwise_mul" ||
             op_type == "fusion_elementwise_mul_activation") {
    elementwise_operation = converter->AddOperation(NNADAPTER_MUL);
  } else if (op_type == "elementwise_div" ||
             op_type == "fusion_elementwise_div_activation") {
    elementwise_operation = converter->AddOperation(NNADAPTER_DIV);
  } else {
    LOG(WARNING) << "Unsupported elementwise op type: " << op_type;
    return FAILED;
  }
  converter->SetOperation(
      elementwise_operation, &input_operands, &output_operands);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    elementwise_add,
    kNNAdapter,
    paddle::lite::subgraph::nnadapter::ElementwiseConverter);
REGISTER_SUBGRAPH_BRIDGE(
    elementwise_sub,
    kNNAdapter,
    paddle::lite::subgraph::nnadapter::ElementwiseConverter);
REGISTER_SUBGRAPH_BRIDGE(
    elementwise_mul,
    kNNAdapter,
    paddle::lite::subgraph::nnadapter::ElementwiseConverter);
REGISTER_SUBGRAPH_BRIDGE(
    elementwise_div,
    kNNAdapter,
    paddle::lite::subgraph::nnadapter::ElementwiseConverter);
REGISTER_SUBGRAPH_BRIDGE(
    fusion_elementwise_add_activation,
    kNNAdapter,
    paddle::lite::subgraph::nnadapter::ElementwiseConverter);
REGISTER_SUBGRAPH_BRIDGE(
    fusion_elementwise_sub_activation,
    kNNAdapter,
    paddle::lite::subgraph::nnadapter::ElementwiseConverter);
REGISTER_SUBGRAPH_BRIDGE(
    fusion_elementwise_mul_activation,
    kNNAdapter,
    paddle::lite::subgraph::nnadapter::ElementwiseConverter);
REGISTER_SUBGRAPH_BRIDGE(
    fusion_elementwise_div_activation,
    kNNAdapter,
    paddle::lite::subgraph::nnadapter::ElementwiseConverter);
