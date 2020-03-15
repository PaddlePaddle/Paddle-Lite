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

#include "lite/kernels/mlu/bridges/graph.h"
#include "lite/kernels/mlu/bridges/utility.h"
#include "lite/kernels/npu/bridges/registry.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

std::vector<int> axis_to_4d(std::vector<int> axis) {
  if (axis.size() >= 4) {
    return axis;
  }
  std::vector<int> new_axis = {0, 1, 2, 3};
  int i = 0;
  for (i = 0; i < axis.size(); i++) {
    new_axis[i] = axis[i];
  }
  return new_axis;
}

int TransposeConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[MLU] Converting " + op_type + "...";

  // Get input vars and op attributes
  auto x_var_name = op_info->Input("X").front();
  // auto x = scope->FindMutableTensor(x_var_name)->GetMutable<Tensor>();
  // auto x_dims = x->dims();

  auto out_var_name = op_info->Output("Out").front();
  auto output = scope->FindVar(out_var_name)->GetMutable<Tensor>();
  auto output_dims = output->dims().Vectorize();

  auto axis = op_info->GetAttr<std::vector<int>>("axis");
  auto axis_4d = axis_to_4d(axis);
  auto output_tensor = graph->AddNode(
      out_var_name, output_dims, CNML_TENSOR, CNML_NHWC, graph->FPType());

  CHECK(graph->HasNode(x_var_name));
  auto input_tensor = graph->GetNode(x_var_name);
  cnmlBaseOp_t transpose_op_{nullptr};

  cnmlNdTransposeOpParam_t transpose_param{nullptr};

  CNML_CALL(cnmlCreateNdTransposeOpParam(
      &transpose_param, axis_4d.data(), axis_4d.size()));

  // Use cnmlCreatexxxOpForward to create op.
  CNML_CALL(cnmlCreateNdTransposeProOp(&transpose_op_,
                                       input_tensor->mlu_tensor(),
                                       output_tensor->mlu_tensor(),
                                       transpose_param));

  graph->FuseOp(transpose_op_);
  return SUCCESS;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
REGISTER_SUBGRAPH_BRIDGE(transpose,
                         kMLU,
                         paddle::lite::subgraph::mlu::TransposeConverter);

REGISTER_SUBGRAPH_BRIDGE(transpose2,
                         kMLU,
                         paddle::lite::subgraph::mlu::TransposeConverter);
