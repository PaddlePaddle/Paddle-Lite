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

int ReduceMeanConverter(void* ctx, OpLite* op, KernelBase* kernel) {
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

  // 2. prepare input2: dimension node
  auto dim = op_info->GetAttr<std::vector<int>>("dim");
  auto axes_node = graph->Add(x_name + "/axes", dim);

  // 3. prepare output:
  auto out_name = op_info->Output("Out").front();
  auto reduce_mean_node = graph->Add<ge::op::ReduceMean>(out_name);

  // 4. deal ascend unsupport attributes
  bool reduce_all = false;
  if (op_info->HasAttr("reduce_all")) {
    reduce_all = op_info->GetAttr<bool>("reduce_all");
  }
  if (reduce_all) {
    LOG(WARNING)
        << "[HUAWEI_ASCEND_NPU] Attr[reduce_all]=true doesn't support!";
    return FAILED;
  }
  // 5. deal ascend need attributes
  // 5.1 keep_dim
  bool keep_dim = false;
  if (op_info->HasAttr("keep_dim")) {
    keep_dim = op_info->GetAttr<bool>("keep_dim");
  }
  VLOG(3) << "[HUAWEI_ASCEND_NPU] keep_dim:" << keep_dim;

  // 6. pack op
  auto reduce_mean_op = reduce_mean_node->data<ge::op::ReduceMean>();
  reduce_mean_op->set_input_x(*x_node->data());
  reduce_mean_op->set_input_axes(*axes_node->data());
  reduce_mean_op->set_attr_keep_dims(keep_dim);
  INPUT_UPDATE(reduce_mean_op, x, x_node);
  INPUT_UPDATE(reduce_mean_op, axes, axes_node);
  OUTPUT_UPDATE(reduce_mean_op, y, reduce_mean_node);

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace huawei_ascend_npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    reduce_mean,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::ReduceMeanConverter);
