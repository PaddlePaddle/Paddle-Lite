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

int DropoutConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Converting " + op_type + "...";

  // Get input, output and op attributes
  auto x_name = op_info->Input("X").front();
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();

  auto out_name = op_info->Output("Out").front();

  auto dropout_implementation =
      op_info->GetAttr<std::string>("dropout_implementation");
  auto scale = 1 - op_info->GetAttr<float>("dropout_prob");
  if (dropout_implementation == "upscale_in_train") {
    scale = 1.f;
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] Huawei Ascend NPU not support "
                    "upscale_in_train implementation!";
    return FAILED;
  }

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x);
  }

  auto input_scale_node =
      graph->Add<float>(x_name + "/scale", scale, x_dims.Vectorize());

  auto dropout_node = graph->Add<ge::op::Scale>(out_name);
  auto dropout_op = dropout_node->data<ge::op::Scale>();
  dropout_op->set_input_x(*x_node->data());
  dropout_op->set_input_scale(*input_scale_node->data());
  dropout_op->set_attr_axis(0);
  dropout_op->set_attr_num_axes(-1);
  dropout_op->set_attr_scale_from_blob(true);
  INPUT_UPDATE(dropout_op, x, x_node);
  INPUT_UPDATE(dropout_op, scale, input_scale_node);
  OUTPUT_UPDATE(dropout_op, y, dropout_node);

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace huawei_ascend_npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    dropout,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::DropoutConverter);
