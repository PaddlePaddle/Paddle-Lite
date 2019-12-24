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

#include "lite/kernels/npu/bridges/registry.h"
#include "lite/kernels/xpu/bridges/graph.h"
#include "lite/kernels/xpu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace xpu {

int DropoutConverter(void* ctx, OpLite* op) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  VLOG(3) << "[XPU] Converting " + op_type + "...";

  // Create node and set params from op
  auto x_var_name = op_info->Input("X").front();
  auto out_var_name = op_info->Output("Out").front();
  auto dropout_prob = op_info->GetAttr<float>("dropout_prob");
  auto dropout_implementation =
      op_info->GetAttr<std::string>("dropout_implementation");
  double rate;
  if (dropout_implementation == "downgrade_in_infer") {
    rate = 1. - dropout_prob;
  } else if (dropout_implementation == "upscale_in_train") {
    rate = 1.;
  } else {
    LOG(FATAL) << "unsupported dropout_implementation == "
               << dropout_implementation << " for dropout";
  }
  CHECK(graph->HasNode(x_var_name));
  graph->AddNode(
      out_var_name,
      graph->builder_.CreateDropout(*graph->GetNode(x_var_name), rate));

  return SUCCESS;
}

}  // namespace xpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(XPU,
                         dropout,
                         paddle::lite::subgraph::xpu::DropoutConverter);
