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

#include "lite/core/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/npu/bridges/graph.h"
#include "lite/kernels/npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace npu {

int IncrementConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NPU] Converting " + op_type + "...";

  // Get input, output and op attributes
  auto x_name = op_info->Input("X").front();
  auto x = scope->FindTensor(x_name);
  auto x_dims = x->dims();

  auto out_name = op_info->Output("Out").front();

  float step = op_info->GetAttr<float>("step");

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x, CvtShape(x_dims));
  }

  // Y node
  Tensor y;
  y.Resize({1});
  auto y_data = y.mutable_data<float>();
  y_data[0] = step;
  y.set_persistable(true);
  auto y_node = graph->Add(out_name + "/y", y);

  // add node
  auto increment_node = graph->Add<ge::op::Add>(out_name);
  auto increment_op = increment_node->data<ge::op::Add>();
  increment_op->set_input_x1(*x_node->data());
  increment_op->set_input_x2(*y_node->data());

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(increment,
                         kNPU,
                         paddle::lite::subgraph::npu::IncrementConverter);
