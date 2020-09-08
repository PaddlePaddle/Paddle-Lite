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

int LrnConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[MLU] Converting " + op_type + "...";

  // Create lrn node and get params from op
  auto fp_type = graph->FPType();
  auto x_var_name = op_info->Input("X").front();
  auto out_var_name = op_info->Output("Out").front();
  auto output = scope->FindVar(out_var_name)->GetMutable<Tensor>();
  auto output_dims = output->dims().Vectorize();
  auto output_tensor = graph->AddNode(
      out_var_name, output_dims, CNML_TENSOR, CNML_NCHW, fp_type);
  CHECK(graph->HasNode(x_var_name));
  auto input_tensor = graph->GetNode(x_var_name);

  auto alpha = op_info->GetAttr<float>("alpha");
  auto beta = op_info->GetAttr<float>("beta");
  auto k = op_info->GetAttr<float>("k");
  if (op_info->HasAttr("norm_region")) {
    CHECK(op_info->GetAttr<std::string>("norm_region") == "AcrossChannels")
        << "Unsuport WithinChannel";
  }
  auto local_size = op_info->GetAttr<int>("n");
  auto input_scale = op_info->GetInputScale(x_var_name)[0];
  VLOG(5) << "lrn input scale: " << input_scale;

  cnmlLrnOpParam_t param;
  cnmlBaseOp_t lrn_op;
  CNML_CALL(
      cnmlCreateLrnOpParam(&param, CNML_LRN_V3, local_size, alpha, beta, k));
  CNML_CALL(cnmlCreateLrnOp(
      &lrn_op, param, input_tensor->mlu_tensor(), output_tensor->mlu_tensor()));
  CNML_CALL(cnmlDestroyLrnOpParam(&param));

  graph->SetComputingDataType(
      lrn_op, input_tensor->mlu_tensor(), 1 / input_scale);
  CNML_CALL(cnmlSetOperationComputingDataType(
      lrn_op, output_tensor->mlu_tensor(), fp_type, nullptr));

  graph->FuseOp(lrn_op);
  CNML_CALL(cnmlDestroyBaseOp(&lrn_op));
  return SUCCESS;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(lrn, kMLU, paddle::lite::subgraph::mlu::LrnConverter);
