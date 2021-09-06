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

#include "lite/core/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/nnadapter/bridges/converter.h"
#include "lite/kernels/nnadapter/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace nnadapter {

DDim concat_infer_shape(const std::vector<const Tensor*>& inputs, int in_axis) {
  std::vector<DDim> input_dims;
  for (auto* tensor : inputs) {
    input_dims.push_back(tensor->dims());
  }
  size_t axis = static_cast<size_t>(in_axis);

  DDim out_dims = input_dims[0];
  for (size_t i = 1; i < input_dims.size(); i++) {
    for (size_t j = 0; j < input_dims[0].size(); j++) {
      if (j == axis) {
        out_dims[axis] += input_dims[i][j];
      } else {
        if (out_dims[j] != input_dims[i][j]) {
          LOG(FATAL) << "infer shape error.";
        }
      }
    }
  }
  if (out_dims[axis] < 0) {
    out_dims[axis] = -1;
  }

  return out_dims;
}

NNAdapterOperand* ConcatOperands(Converter* converter,
                                 Scope* scope,
                                 const std::vector<std::string>& input_names,
                                 const DDim& output_dims,
                                 const std::string& output_name,
                                 int axis) {
  NNAdapterOperand* output_operand = nullptr;
  if (converter->HasOperand(output_name)) {
    output_operand = converter->GetOperand(output_name);
  } else {
    // Concat inputs
    std::vector<NNAdapterOperand*> input_operands;
    for (size_t i = 0; i < input_names.size(); i++) {
      auto input_name = input_names[i];
      auto input = scope->FindMutableTensor(input_name);
      auto input_dims = input->dims();
      NNAdapterOperand* input_operand = nullptr;
      if (converter->HasOperand(input_name)) {
        input_operand = converter->GetOperand(input_name);
      } else {
        input_operand =
            converter->AddFloat32VariableOperand(input_dims, input_name);
      }
      input_operands.push_back(input_operand);
    }
    // Concat axis
    auto* axis_operand = converter->AddInt32ConstantOperand(axis);
    input_operands.push_back(axis_operand);
    // Concat output
    output_operand = converter->AddFloat32VariableOperand(
        DDim({static_cast<int64_t>(input_names.size())}), output_name);
    std::vector<NNAdapterOperand*> output_operands = {output_operand};
    converter->AddOperation(
        NNADAPTER_CONCAT, &input_operands, &output_operands);
  }
  return output_operand;
}

NNAdapterOperand* ReshapeOperands(Converter* converter,
                                  NNAdapterOperand* input_operand,
                                  std::vector<int64_t> shape_data,
                                  const std::string& output_name) {
  NNAdapterOperand* output_operand = nullptr;
  if (converter->HasOperand(output_name)) {
    output_operand = converter->GetOperand(output_name);
  } else {
    // Reshape inputs
    std::vector<NNAdapterOperand*> input_operands;
    input_operands.push_back(input_operand);
    // Reshape shape
    std::vector<int32_t> shape_data_int32;
    for (auto ele : shape_data) {
      shape_data_int32.push_back(static_cast<int32_t>(ele));
    }
    auto shape_operand = converter->AddInt32ConstantOperand(
        &shape_data_int32[0],
        DDim({static_cast<int64_t>(shape_data_int32.size())}));
    input_operands.push_back(shape_operand);
    // Reshape output
    output_operand =
        converter->AddFloat32VariableOperand(DDim(shape_data), output_name);
    std::vector<NNAdapterOperand*> output_operands = {output_operand};
    converter->AddOperation(
        NNADAPTER_RESHAPE, &input_operands, &output_operands);
  }
  return output_operand;
}

int StackConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto converter = static_cast<Converter*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "Converting " << op_type << " ...";

  // Get input and output vars and op attributes
  auto x_names = op_info->Input("X");
  auto x_num = x_names.size();
  auto x0 = scope->FindMutableTensor(x_names[0]);
  auto x0_dims = x0->dims();
  auto x0_rank = x0_dims.size();
  auto output_name = op_info->Output("Y").front();
  auto axis = op_info->GetAttr<int>("axis");
  if (axis < 0) {
    axis += (x0_rank + 1);
  }
  std::vector<const Tensor*> x_tensor_list;
  for (auto x_name : x_names) {
    x_tensor_list.push_back(scope->FindMutableTensor(x_name));
  }
  auto concat_axis = axis;
  if (axis == x0_rank) {
    concat_axis = axis - 1;
  }
  auto output_dims = concat_infer_shape(x_tensor_list, concat_axis);

  // Concat input operand
  std::string concat_name = output_name + "/concat";
  auto input_concat_operand = ConcatOperands(
      converter, scope, x_names, output_dims, concat_name, concat_axis);

  // Reshape output operand
  std::vector<int64_t> shape_data(x0_dims.Vectorize());
  shape_data.insert(shape_data.begin() + axis, x_num);
  ReshapeOperands(converter, input_concat_operand, shape_data, output_name);

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(stack,
                         kNNAdapter,
                         paddle::lite::subgraph::nnadapter::StackConverter);
