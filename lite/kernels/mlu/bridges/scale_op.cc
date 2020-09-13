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

int ScaleConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[MLU] Converting " + op_type + "...";

  // Create act node and set params from op
  auto x_var_name = op_info->Input("X").front();
  auto out_var_name = op_info->Output("Out").front();
  auto output = scope->FindVar(out_var_name)->GetMutable<Tensor>();
  auto output_dims = output->dims().Vectorize();
  auto output_tensor = graph->AddNode(
      out_var_name, output_dims, CNML_TENSOR, CNML_NCHW, graph->FPType());
  auto bias_after_scale = op_info->GetAttr<bool>("bias_after_scale");
  auto scale = op_info->GetAttr<float>("scale");
  auto bias = op_info->GetAttr<float>("bias");
  auto beta = bias_after_scale ? bias : bias * scale;

  std::vector<int64_t> shape = {1, 1, 1, 1};

  std::string prefix = string_format("_%p", op);
  auto alpha_tensor = graph->AddNode(
      "Alpha" + prefix, shape, CNML_CONST, CNML_NHWC, graph->FPType());
  auto beta_tensor = graph->AddNode(
      "Beta" + prefix, shape, CNML_CONST, CNML_NHWC, graph->FPType());

  graph->BindConstRawData("Alpha" + prefix, &scale, 1);
  graph->BindConstRawData("Beta" + prefix, &beta, 1);

  auto input_tensor = graph->GetNode(x_var_name);
  cnmlBaseOp_t scale_op;
  CNML_CALL(cnmlCreateScaleOp(&scale_op,
                              input_tensor->mlu_tensor(),
                              output_tensor->mlu_tensor(),
                              alpha_tensor->mlu_tensor(),
                              beta_tensor->mlu_tensor()));
  graph->FuseOp(scale_op);
  CNML_CALL(cnmlDestroyBaseOp(&scale_op));
  return SUCCESS;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(scale,
                         kMLU,
                         paddle::lite::subgraph::mlu::ScaleConverter);
