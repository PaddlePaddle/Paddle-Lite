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

#include "lite/core/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/nnadapter/bridges/converter.h"
#include "lite/kernels/nnadapter/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace nnadapter {

int ExpandConverter(void* ctx, OpLite* op, KernelBase* kernel) {
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
  // Shape operand
  NNAdapterOperand* shape_operand = nullptr;
  if (op_info->HasInput("Shape") && !op_info->Input("Shape").empty()) {
    auto shape_name = op_info->Input("Shape").front();
    auto shape_tensor = scope->FindTensor(shape_name);
    if (converter->HasOperand(shape_name)) {
      shape_operand = converter->GetOperand(shape_name);
    } else {
      shape_operand = converter->AddOperand(shape_tensor, shape_name);
    }
  } else if (op_info->HasInput("expand_shapes_tensor") &&
             !op_info->Input("expand_shapes_tensor").empty()) {
    LOG(ERROR) << "Not support expand_shapes_tensor now.";
  } else {
    std::vector<int> expand_shape = op_info->GetAttr<std::vector<int>>("shape");
    auto vec_in_dims = x_dims.Vectorize();
    auto diff = expand_shape.size() - vec_in_dims.size();
    vec_in_dims.insert(vec_in_dims.begin(), diff, 1);
    std::vector<int> final_expand_shape(vec_in_dims.size());
    for (size_t i = 0; i < vec_in_dims.size(); ++i) {
      if (i < diff) {  // expand_shape = [3,4,-1,-1], X = [10,2] -->
                       // final_expand_shape = [3,4,10,2]
        final_expand_shape[i] = expand_shape[i];
      } else if (expand_shape[i] > 0) {  // expand_shape = [3,4,10,4], X =
                                         // [10,1] --> final_expand_shape =
                                         // [3,4,10,4]
        if (vec_in_dims[i] != 1) {
          final_expand_shape[i] = expand_shape[i];
        } else {
          final_expand_shape[i] = expand_shape[i];
        }
      } else {  // expand_shape = [3,4,-1,-1], X = [10,2] --> final_expand_shape
                // = [3,4,10,2]
        final_expand_shape[i] = vec_in_dims[i];
      }
    }
    shape_operand = converter->AddInt32ConstantOperand(
        final_expand_shape.data(),
        DDim({static_cast<int64_t>(final_expand_shape.size())}));
  }
  // Output operand
  NNAdapterOperand* output_operand = nullptr;
  if (has_out_scale) {
    output_operand =
        converter->AddQuant8VariableOperand(out_dims, out_scale, out_name);
  } else {
    output_operand = converter->AddFloat32VariableOperand(out_dims, out_name);
  }

  // Expand operation
  std::vector<NNAdapterOperand*> input_operands = {input_operand,
                                                   shape_operand};
  std::vector<NNAdapterOperand*> output_operands = {output_operand};
  converter->AddOperation(NNADAPTER_EXPAND, &input_operands, &output_operands);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(expand_v2,
                         kNNAdapter,
                         paddle::lite::subgraph::nnadapter::ExpandConverter);
