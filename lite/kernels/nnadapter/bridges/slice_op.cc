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

int SliceConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto converter = static_cast<Converter*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "Converting " << op_type << " ...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("Input").front();
  auto x_scale_name = "Input0_scale";
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
  std::vector<int> axes = op_info->GetAttr<std::vector<int>>("axes");
  std::vector<int> starts = op_info->GetAttr<std::vector<int>>("starts");
  std::vector<int> ends_ori = op_info->GetAttr<std::vector<int>>("ends");
  // paddle model: ends[i] maybe is bigger than x_dims[axes[i]], so it needs to
  // update.
  int axes_size = static_cast<int>(axes.size());
  std::vector<int> ends(axes_size, 0);
  for (int i = 0; i < axes_size; i++) {
    ends[i] = ends_ori[i] > x_dims[axes[i]] ? x_dims[axes[i]] : ends_ori[i];
  }

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
  // Axes operand
  NNAdapterOperand* axes_operand = converter->AddInt32ConstantOperand(
      &axes[0], DDim({static_cast<int64_t>(axes.size())}));
  // Starts operand
  NNAdapterOperand* starts_operand = converter->AddInt32ConstantOperand(
      &starts[0], DDim({static_cast<int64_t>(starts.size())}));
  // Ends operand
  NNAdapterOperand* ends_operand = converter->AddInt32ConstantOperand(
      &ends[0], DDim({static_cast<int64_t>(ends.size())}));
  // Steps operand
  std::vector<int> steps(axes_size, 1);
  NNAdapterOperand* steps_operand = converter->AddInt32ConstantOperand(
      &steps[0], DDim({static_cast<int64_t>(steps.size())}));
  // Output operand
  NNAdapterOperand* output_operand = nullptr;
  if (has_out_scale) {
    output_operand =
        converter->AddQuant8VariableOperand(out_dims, out_scale, out_name);
  } else {
    output_operand = converter->AddFloat32VariableOperand(out_dims, out_name);
  }

  // slice operation
  std::vector<NNAdapterOperand*> input_operands = {
      input_operand, axes_operand, starts_operand, ends_operand, steps_operand};
  std::vector<NNAdapterOperand*> output_operands = {output_operand};
  NNAdapterOperation* slice_operation =
      converter->AddOperation(NNADAPTER_SLICE);
  converter->SetOperation(slice_operation, &input_operands, &output_operands);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(slice,
                         kNNAdapter,
                         paddle::lite::subgraph::nnadapter::SliceConverter);
