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

int ClipConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Converting " + op_type + "...";

  float minValue =
      op_info->HasAttr("min") ? op_info->GetAttr<float>("min") : 0.0f;
  float maxValue =
      op_info->HasAttr("max") ? op_info->GetAttr<float>("max") : 0.0f;

  // X node
  auto x_name = op_info->Input("X").front();
  auto x_tensor = scope->FindMutableTensor(x_name);
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x_tensor);
  }

  // Min node
  std::shared_ptr<Node> min_node = nullptr;
  if (op_info->HasInput("Min") && op_info->Input("Min").size() > 0) {
    auto min_name = op_info->Input("Min").front();
    auto min_tensor = scope->FindMutableTensor(min_name);
    if (graph->Has(min_name)) {
      min_node = graph->Get(min_name);
    } else {
      min_node = graph->Add(min_name, *min_tensor);
    }
  } else {
    min_node = graph->Add(x_name + "/Min", minValue);
  }

  // Max node
  std::shared_ptr<Node> max_node = nullptr;
  if (op_info->HasInput("Max") && op_info->Input("Max").size() > 0) {
    auto max_name = op_info->Input("Max").front();
    auto max_tensor = scope->FindMutableTensor(max_name);
    if (graph->Has(max_name)) {
      max_node = graph->Get(max_name);
    } else {
      max_node = graph->Add(max_name, *max_tensor);
    }
  } else {
    max_node = graph->Add(x_name + "/Max", maxValue);
  }

  auto out_name = op_info->Output("Out").front();

  // Clip node
  auto clip_node = graph->Add<ge::op::ClipByValue>(out_name);
  auto clip_op = clip_node->data<ge::op::ClipByValue>();
  clip_op->set_input_x(*x_node->data());
  clip_op->set_input_clip_value_min(*min_node->data());
  clip_op->set_input_clip_value_max(*max_node->data());
  INPUT_UPDATE(clip_op, x, x_node);
  INPUT_UPDATE(clip_op, clip_value_min, min_node);
  INPUT_UPDATE(clip_op, clip_value_max, max_node);
  OUTPUT_UPDATE(clip_op, y, clip_node);

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace huawei_ascend_npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    clip,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::ClipConverter);
