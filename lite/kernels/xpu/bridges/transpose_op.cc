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

int TransposeConverter(void* ctx, OpLite* op) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  VLOG(3) << "[XPU] Converting " + op_type + "...";

  // Create node and set params from op
  auto x_var_name = op_info->Input("X").front();
  auto out_var_name = op_info->Output("Out").front();

  auto axis = op_info->GetAttr<std::vector<int>>("axis");

  CHECK(graph->HasNode(x_var_name));
  graph->AddNode(
      out_var_name,
      graph->builder_.CreateTranspose(
          *graph->GetNode(x_var_name),
          Cvt2ArrayInt(std::vector<int64_t>(axis.begin(), axis.end()))));

  return SUCCESS;
}

}  // namespace xpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(XPU,
                         transpose,
                         paddle::lite::subgraph::xpu::TransposeConverter);
REGISTER_SUBGRAPH_BRIDGE(XPU,
                         transpose2,
                         paddle::lite::subgraph::xpu::TransposeConverter);
