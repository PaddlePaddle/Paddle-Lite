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

int ActConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[MLU] Converting " + op_type + "...";

  // Create act node and set params from op
  auto fp_type = graph->FPType();
  auto x_var_name = op_info->Input("X").front();
  auto out_var_name = op_info->Output("Out").front();
  auto output = scope->FindVar(out_var_name)->GetMutable<Tensor>();
  auto output_dims = output->dims().Vectorize();
  auto output_tensor = graph->AddNode(
      out_var_name, output_dims, CNML_TENSOR, CNML_NCHW, fp_type);
  CHECK(graph->HasNode(x_var_name));
  auto input_tensor = graph->GetNode(x_var_name);
  cnmlBaseOp_t activation_op;
  if (op_type == "leaky_relu") {
    auto alpha = op_info->GetAttr<float>("alpha");
    std::vector<int64_t> shape = {1, 1, 1, 1};
    std::string alpha_var_name = string_format("leaky_relu_alpha_%p", op);
    auto alpha_tensor =
        graph->AddNode(alpha_var_name, shape, CNML_CONST, CNML_NHWC, fp_type);
    graph->BindConstRawData(alpha_var_name, &alpha, 1, true);
    CNML_CALL(cnmlCreatePreluOp(&activation_op,
                                input_tensor->mlu_tensor(),
                                output_tensor->mlu_tensor(),
                                alpha_tensor->mlu_tensor()));
  } else {
    cnmlActiveFunction_t act_type = OpTypeToCNMLActType(op_type);
    CNML_CALL(cnmlCreateActiveOp(&activation_op,
                                 act_type,
                                 input_tensor->mlu_tensor(),
                                 output_tensor->mlu_tensor()));
  }
  graph->FuseOp(activation_op);
  CNML_CALL(cnmlDestroyBaseOp(&activation_op));
  return SUCCESS;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(sigmoid,
                         kMLU,
                         paddle::lite::subgraph::mlu::ActConverter);
REGISTER_SUBGRAPH_BRIDGE(relu, kMLU, paddle::lite::subgraph::mlu::ActConverter);
REGISTER_SUBGRAPH_BRIDGE(relu6,
                         kMLU,
                         paddle::lite::subgraph::mlu::ActConverter);
REGISTER_SUBGRAPH_BRIDGE(tanh, kMLU, paddle::lite::subgraph::mlu::ActConverter);
REGISTER_SUBGRAPH_BRIDGE(leaky_relu,
                         kMLU,
                         paddle::lite::subgraph::mlu::ActConverter);
