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

int UnsqueezeConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Converting " + op_type + "...";

  // Get input, output and op attributes
  // 1. prepare input1: X node
  auto x_name = op_info->Input("X").front();
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();

  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x, CvtShape(x_dims));
  }

  // 2. prepare output:
  auto out_name = op_info->Output("Out").front();
  auto unsqueeze_node = graph->Add<ge::op::Unsqueeze>(out_name);

  // 3. Deal paddle's param that ascend is not suport.
  if ((op_info->HasInput("AxesTensor") &&
       op_info->Input("AxesTensor").size() > 0) ||
      (op_info->HasInput("AxesTensorList") &&
       op_info->Input("AxesTensorList").size() > 0)) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] doesn't support AxesTensor";
    return FAILED;
  }

  std::vector<int> axes{};
  // 3. prepare ascend need attributes
  if (op_info->HasAttr("axes")) {
    axes = op_info->GetAttr<std::vector<int>>("axes");
  }

  // 4. pack op
  auto unsqueeze_op = unsqueeze_node->data<ge::op::Unsqueeze>();
  unsqueeze_op->set_input_x(*x_node->data());
  unsqueeze_op->set_attr_axes(
      ge::Operator::OpListInt(axes.begin(), axes.end()));

  INPUT_UPDATE(unsqueeze_op, x, x_node);
  OUTPUT_UPDATE(unsqueeze_op, y, unsqueeze_node);

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace huawei_ascend_npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    unsqueeze,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::UnsqueezeConverter);

REGISTER_SUBGRAPH_BRIDGE(
    unsqueeze2,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::UnsqueezeConverter);
