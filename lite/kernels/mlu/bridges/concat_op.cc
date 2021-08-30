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
#include "lite/kernels/mlu/bridges/graph.h"
#include "lite/kernels/mlu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

int ConcatConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[MLU] Converting " + op_type + "...";

  auto x_var_name = op_info->Input("X");
  auto out_var_name = op_info->Output("Out").front();
  auto output = scope->FindVar(out_var_name)->GetMutable<Tensor>();
  auto output_dims = output->dims().Vectorize();
  auto param_axis = op_info->GetAttr<int>("axis");

  std::vector<cnmlTensor_t> input_tensor;
  for (auto x_name : x_var_name) {
    CHECK(graph->HasNode(x_name));
    input_tensor.push_back(graph->GetNode(x_name)->mlu_tensor());
  }

  auto dims = output_dims.size();
  int axis = (param_axis < 0) ? (param_axis + dims) : param_axis;
  CHECK_LT(axis, dims) << "Unsupport dims in mlu concat";
  // value of nhwc2nchw_axis is index of nhwc
  // order of nhwc2nchw_axis is nchw
  int nhwc_axis = GetAxisNHWC2NCHW<int>(dims)[axis];

  auto output_tensor = graph->AddNode(
      out_var_name, output_dims, CNML_TENSOR, CNML_NCHW, graph->FPType());

  cnmlBaseOp_t concat_op;
  cnmlTensor_t outputs = output_tensor->mlu_tensor();
  CNML_CALL(cnmlCreateNdConcatOp(&concat_op,
                                 nhwc_axis,
                                 input_tensor.data(),
                                 x_var_name.size(),
                                 &outputs,
                                 1));
  graph->FuseOp(concat_op);
  CNML_CALL(cnmlDestroyBaseOp(&concat_op));
  return SUCCESS;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(concat,
                         kMLU,
                         paddle::lite::subgraph::mlu::ConcatConverter);
