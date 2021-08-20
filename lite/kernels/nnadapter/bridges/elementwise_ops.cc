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
  int x_rank = x_dims.size();
  auto x_persistable = x->persistable();
  auto y_name = op_info->Input("Y").front();
  auto y_scale_name = "Y0_scale";
  auto has_y_scale = op_info->HasInputScale(y_scale_name, true);
  auto y_scale =
      has_y_scale ? op_info->GetInputScale(y_scale_name, true)[0] : 0.f;
  auto y = scope->FindMutableTensor(y_name);
  auto y_dims = y->dims();
  int y_rank = y_dims.size();
  auto y_persistable = y->persistable();
  auto out_name = op_info->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  auto has_out_scale = op_info->HasOutputScale(out_scale_name, true);
  auto out_scale =
      has_out_scale ? op_info->GetOutputScale(out_scale_name, true)[0] : 0.f;
  auto out = scope->FindMutableTensor(out_name);
  auto out_dims = out->dims();
  auto axis = op_info->GetAttr<int>("axis");
  if (axis == -1) {
    axis = std::abs(x_rank - y_rank);
  }
  auto act_type = op_info->HasAttr("act_type")
                      ? op_info->GetAttr<std::string>("act_type")
                      : "";
  // Check whether the two dimensions are compatiable(Numpy-style broadcasting
  // https://numpy.org/doc/stable/user/basics.broadcasting.html).
  // Fill the dimension of Y with 1
  int max_rank = x_rank > y_rank ? x_rank : y_rank;
  std::vector<int64_t> x_shape(max_rank, 1);
  std::vector<int64_t> y_shape(max_rank, 1);
  if (x_rank > y_rank) {
    for (int i = 0; i < y_rank; i++) {
      y_shape[i + axis] = y_dims[i];
    }
    y_dims = DDim(y_shape);
  } else {
    for (int i = 0; i < x_rank; i++) {
      x_shape[i + axis] = x_dims[i];
    }
    x_dims = DDim(x_shape);
  }
  bool matched = true;
  for (int i = 0; i < max_rank; i++) {
    matched &= (x_dims[i] == y_dims[i] || x_dims[i] == 1 || y_dims[i] == 1);
  }
  CHECK(matched) << "Incompatible broadcasting for x " << x->dims().repr()
                 << " y " << y->dims().repr();

  // Input0 operand
  NNAdapterOperand* input0_operand = nullptr;
  if (converter->HasOperand(x_name)) {
    input0_operand = converter->GetOperand(x_name);
    // TODO(hong19860320) Add a NNADAPTER_RESHAPE operation if x_dims is changed
  } else {
    if (has_x_scale) {
      input0_operand =
          x_persistable
              ? converter->AddQuant8ConstantOperand(
                    reinterpret_cast<int8_t*>(x->raw_data()), x_dims, x_scale)
              : converter->AddQuant8VariableOperand(x_dims, x_scale, x_name);
    } else {
      input0_operand =
          x_persistable
              ? converter->AddFloat32ConstantOperand(
                    reinterpret_cast<float*>(x->raw_data()), x_dims)
              : converter->AddFloat32VariableOperand(x_dims, x_name);
    }
  }

  // Input1 operand
  NNAdapterOperand* input1_operand = nullptr;
  if (converter->HasOperand(y_name)) {
    input1_operand = converter->GetOperand(y_name);
    // TODO(hong19860320) Add a NNADAPTER_RESHAPE operation if y_dims is changed
  } else {
    if (has_y_scale) {
      input1_operand =
          y_persistable
              ? converter->AddQuant8ConstantOperand(
                    reinterpret_cast<int8_t*>(y->raw_data()), y_dims, y_scale)
              : converter->AddQuant8VariableOperand(y_dims, y_scale, y_name);
    } else {
      input1_operand =
          y_persistable
              ? converter->AddFloat32ConstantOperand(
                    reinterpret_cast<float*>(y->raw_data()), y_dims)
              : converter->AddFloat32VariableOperand(y_dims, y_name);
    }
  }

  // Fuse code operand
  int32_t fuse_code_value = NNADAPTER_FUSED_NONE;
  if (act_type == "relu") {
    fuse_code_value = NNADAPTER_FUSED_RELU;
  } else if (act_type == "relu1") {
    fuse_code_value = NNADAPTER_FUSED_RELU1;
  } else if (act_type == "relu6") {
    fuse_code_value = NNADAPTER_FUSED_RELU6;
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

  // ADD, SUB, MUL, DIV, MAX and MIN operation
  std::vector<NNAdapterOperand*> input_operands = {
      input0_operand, input1_operand, fuse_code_operand};
  std::vector<NNAdapterOperand*> output_operands = {output_operand};
  NNAdapterOperation* elementwise_operation = nullptr;
  NNAdapterOperationType eltwise_operation_type;
  if (op_type == "elementwise_add" ||
      op_type == "fusion_elementwise_add_activation") {
    eltwise_operation_type = NNADAPTER_ADD;
  } else if (op_type == "elementwise_sub" ||
             op_type == "fusion_elementwise_sub_activation") {
    eltwise_operation_type = NNADAPTER_SUB;
  } else if (op_type == "elementwise_mul" ||
             op_type == "fusion_elementwise_mul_activation") {
    eltwise_operation_type = NNADAPTER_MUL;
  } else if (op_type == "elementwise_div" ||
             op_type == "fusion_elementwise_div_activation") {
    eltwise_operation_type = NNADAPTER_DIV;
  } else if (op_type == "elementwise_max" ||
             op_type == "fusion_elementwise_max_activation") {
    eltwise_operation_type = NNADAPTER_MAX;
  } else if (op_type == "elementwise_min" ||
             op_type == "fusion_elementwise_min_activation") {
    eltwise_operation_type = NNADAPTER_MIN;
  } else {
    LOG(WARNING) << "Unsupported elementwise op type: " << op_type;
    return FAILED;
  }
  converter->AddOperation(
      eltwise_operation_type, &input_operands, &output_operands);
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
    elementwise_max,
    kNNAdapter,
    paddle::lite::subgraph::nnadapter::ElementwiseConverter);
REGISTER_SUBGRAPH_BRIDGE(
    elementwise_min,
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
REGISTER_SUBGRAPH_BRIDGE(
    fusion_elementwise_min_activation,
    kNNAdapter,
    paddle::lite::subgraph::nnadapter::ElementwiseConverter);
REGISTER_SUBGRAPH_BRIDGE(
    fusion_elementwise_max_activation,
    kNNAdapter,
    paddle::lite::subgraph::nnadapter::ElementwiseConverter);
