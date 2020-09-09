// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include "lite/kernels/huawei_ascend_npu/bridges/graph.h"
#include "lite/kernels/huawei_ascend_npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace huawei_ascend_npu {

int TransposeConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Converting " + op_type + "...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();

  auto out_name = op_info->Output("Out").front();

  auto axis = op_info->GetAttr<std::vector<int>>("axis");

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x);
  }

  // Transpose node
  auto transpose_node = graph->Add<ge::op::TransposeD>(out_name);
  auto transpose_op = transpose_node->data<ge::op::TransposeD>();
  transpose_op->set_input_x(*x_node->data());
  transpose_op->set_attr_perm(
      ge::Operator::OpListInt(axis.begin(), axis.end()));
  INPUT_UPDATE(transpose_op, x, x_node);
  OUTPUT_UPDATE(transpose_op, y, transpose_node);

  return SUCCESS;
}

}  // namespace huawei_ascend_npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    transpose,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::TransposeConverter);
REGISTER_SUBGRAPH_BRIDGE(
    transpose2,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::TransposeConverter);
