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

int ArgmaxConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[MLU] Converting " + op_type + "...";

  // Get input vars and op attributes
  auto x_var_name = op_info->Input("X").front();
  auto x = scope->FindVar(x_var_name)->GetMutable<Tensor>();
  auto x_dims = x->dims().Vectorize();

  auto out_var_name = op_info->Output("Out").front();
  auto output = scope->FindVar(out_var_name)->GetMutable<Tensor>();
  auto output_dims = output->dims().Vectorize();

  int axis = op_info->GetAttr<int64_t>("axis");

  cnmlDimension_t argmax_mode = static_cast<cnmlDimension_t>(axis);
  auto mlu_output_dim = x->dims().Vectorize();
  // shape is NCHW, layout is NHWC
  mlu_output_dim[axis] = 1;
  auto output_tensor = graph->AddNode(
      out_var_name, mlu_output_dim, CNML_TENSOR, CNML_NCHW, graph->FPType());

  CHECK(graph->HasNode(x_var_name));
  auto input_tensor = graph->GetNode(x_var_name);
  cnmlBaseOp_t argmax_op{nullptr};

  CNML_CALL(cnmlCreateArgmaxOp(&argmax_op,
                               argmax_mode,
                               input_tensor->mlu_tensor(),
                               output_tensor->mlu_tensor()));
  graph->FuseOp(argmax_op);
  CNML_CALL(cnmlDestroyBaseOp(&argmax_op));
  return SUCCESS;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(argmax,
                         kMLU,
                         paddle::lite::subgraph::mlu::ArgmaxConverter);
