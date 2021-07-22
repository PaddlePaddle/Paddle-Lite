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

int NormConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Converting " + op_type + "...";

  auto axis = op_info->GetAttr<int>("axis");
  auto epsilon = op_info->GetAttr<float>("epsilon");

  // X node
  auto x_name = op_info->Input("X").front();
  auto x_tensor = scope->FindMutableTensor(x_name);
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x_tensor);
  }

  auto out_name = op_info->Output("Out").front();
  auto l2_norm_node = graph->Add<ge::op::L2Normalize>(out_name);
  auto l2_norm_op = l2_norm_node->data<ge::op::L2Normalize>();
  l2_norm_op->set_input_x(*x_node->data());
  l2_norm_op->set_attr_axis(ge::Operator::OpListInt({axis}));
  l2_norm_op->set_attr_eps(epsilon);
  INPUT_UPDATE(l2_norm_op, x, x_node);
  OUTPUT_UPDATE(l2_norm_op, y, l2_norm_node);

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace huawei_ascend_npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    norm,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::NormConverter);
