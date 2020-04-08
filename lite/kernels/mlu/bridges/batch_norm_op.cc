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

int BatchNormConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[MLU] Converting " + op_type + "...";

  // Get input vars and op attributes
  auto x_var_name = op_info->Input("X").front();
  auto scale_var_name = op_info->Input("Scale").front();
  auto bias_var_name = op_info->Input("Bias").front();
  auto mean_var_name = op_info->Input("Mean").front();
  auto variance_var_name = op_info->Input("Variance").front();
  auto y_var_name = op_info->Output("Y").front();
  auto epsilon = op_info->GetAttr<float>("epsilon");

  auto output = scope->FindVar(y_var_name)->GetMutable<Tensor>();
  auto output_dims = output->dims().Vectorize();
  auto output_tensor = graph->AddNode(
      y_var_name, output_dims, CNML_TENSOR, CNML_NCHW, graph->FPType());

  CHECK(graph->HasNode(x_var_name));

  auto mean = scope->FindVar(mean_var_name)->GetMutable<Tensor>();
  auto mean_dims = mean->dims().Vectorize();
  auto mean_tensor = graph->AddNode(
      mean_var_name, mean_dims, CNML_CONST, CNML_CNHW, graph->FPType());

  auto variance = scope->FindVar(variance_var_name)->GetMutable<Tensor>();
  auto variance_dims = variance->dims().Vectorize();
  auto variance_tensor = graph->AddNode(
      variance_var_name, variance_dims, CNML_CONST, CNML_CNHW, graph->FPType());

  auto scale = scope->FindVar(scale_var_name)->GetMutable<Tensor>();
  auto bias = scope->FindVar(bias_var_name)->GetMutable<Tensor>();

  int co = static_cast<int>(mean_dims[0]);

  for (int i = 0; i < co; ++i) {
    variance->mutable_data<float>()[i] =
        scale->data<float>()[i] / sqrtf(variance->data<float>()[i] + epsilon);
    mean->mutable_data<float>()[i] =
        mean->data<float>()[i] -
        bias->data<float>()[i] / variance->data<float>()[i];
  }

  auto input_tensor = graph->GetNode(x_var_name);
  cnmlBaseOp_t bn_op;
  CNML_CALL(cnmlCreateBatchNormOpForward(&bn_op,
                                         input_tensor->mlu_tensor(),
                                         output_tensor->mlu_tensor(),
                                         mean_tensor->mlu_tensor(),
                                         variance_tensor->mlu_tensor()));

  graph->BindConstData(variance_var_name, variance);
  graph->BindConstData(mean_var_name, mean);
  graph->FuseOp(bn_op);

  CNML_CALL(cnmlDestroyBaseOp(&bn_op));

  return SUCCESS;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(batch_norm,
                         kMLU,
                         paddle::lite::subgraph::mlu::BatchNormConverter);
