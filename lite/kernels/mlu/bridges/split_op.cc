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
#include "lite/kernels/mlu/bridges/graph.h"
#include "lite/kernels/mlu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

int SplitConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[MLU] Converting " + op_type + "...";

  auto x_var_name = op_info->Input("X").front();
  auto x = scope->FindVar(x_var_name)->GetMutable<Tensor>();
  auto x_dims = x->dims().Vectorize();

  auto out_var_name = op_info->Output("Out");

  auto param_axis = op_info->GetAttr<int>("axis");

  auto num = op_info->GetAttr<int>("num");
  auto sections = op_info->GetAttr<std::vector<int>>("sections");
  int64_t sections_num = static_cast<int64_t>(sections.size());
  auto output_num = num > 0 ? num : sections_num;

  std::vector<cnmlTensor_t> output_tensor;
  for (auto out_name : out_var_name) {
    auto out = scope->FindVar(out_name)->GetMutable<Tensor>();
    auto out_dims = out->dims().Vectorize();
    auto out_tensor = graph->AddNode(
        out_name, out_dims, CNML_TENSOR, CNML_NCHW, graph->FPType());
    output_tensor.push_back(out_tensor->mlu_tensor());
  }

  auto dims = x_dims.size();
  int axis = (param_axis < 0) ? (param_axis + dims) : param_axis;
  CHECK_LE(axis, 4) << "Unsupport dims in mlu concat";
  int nhwc_axis = GetAxisNHWC2NCHW<int>(dims)[axis];

  CHECK(graph->HasNode(x_var_name));
  auto input_tensor = graph->GetNode(x_var_name);

  cnmlBaseOp_t split_op;
  cnmlTensor_t inputs = input_tensor->mlu_tensor();
  CNML_CALL(cnmlCreateNdSplitOp(
      &split_op, nhwc_axis, &inputs, 1, output_tensor.data(), output_num));
  graph->FuseOp(split_op);
  CNML_CALL(cnmlDestroyBaseOp(&split_op));
  return SUCCESS;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(split,
                         kMLU,
                         paddle::lite::subgraph::mlu::SplitConverter);
