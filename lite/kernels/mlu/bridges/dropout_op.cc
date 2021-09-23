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

int DropoutConverter(void* ctx, OpLite* op, KernelBase* kernel) {
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
  /* auto mask_var_name = op_info->Output("Mask").front(); */
  auto output = scope->FindVar(out_var_name)->GetMutable<Tensor>();
  auto output_dims = output->dims().Vectorize();
  auto output_tensor = graph->AddNode(
      out_var_name, output_dims, CNML_TENSOR, CNML_NCHW, graph->FPType());
  /* auto mask = scope->FindVar(mask_var_name)->GetMutable<Tensor>(); */
  /* auto mask_dims = mask->dims().Vectorize(); */
  /* auto mask_tensor = graph->AddNode( */
  /*     mask_var_name, mask_dims, CNML_TENSOR, CNML_NCHW, graph->FPType()); */

  // is_test is true by default
  // if(op_info->HasAttr("is_test")){
  //   auto is_test = op_info->GetAttr<bool>("is_test");
  //   CHECK(is_test != true);
  // }

  // Param fix_seed and seed is useless in MLU

  auto dropout_implementation =
      op_info->GetAttr<std::string>("dropout_implementation");
  auto dropout_prob = op_info->GetAttr<float>("dropout_prob");
  float alpha = 1.0f - dropout_prob;
  if (dropout_implementation == "upscale_in_train") {
    alpha = 1.;
  }
  float beta = 0.;

  std::vector<int64_t> shape = {1, 1, 1, 1};
  std::string alpha_var_name = string_format("dropout_alpha_%p", op);
  std::string beta_var_name = string_format("dropout_beta_%p", op);
  auto alpha_tensor = graph->AddNode(
      alpha_var_name, shape, CNML_CONST, CNML_NHWC, graph->FPType());
  auto beta_tensor = graph->AddNode(
      beta_var_name, shape, CNML_CONST, CNML_NHWC, graph->FPType());

  graph->BindConstRawData(alpha_var_name, &alpha, 1);
  graph->BindConstRawData(beta_var_name, &beta, 1);

  auto input_tensor = graph->GetNode(x_var_name);
  cnmlBaseOp_t scale_op;
  CNML_CALL(cnmlCreateScaleOp(&scale_op,
                              input_tensor->mlu_tensor(),
                              output_tensor->mlu_tensor(),
                              alpha_tensor->mlu_tensor(),
                              beta_tensor->mlu_tensor()));
  graph->FuseOp(scale_op);
  return SUCCESS;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(dropout,
                         kMLU,
                         paddle::lite::subgraph::mlu::DropoutConverter);
