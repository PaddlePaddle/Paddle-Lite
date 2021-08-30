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

int FillConstantBatchSizeLikeConverter(void* ctx,
                                       OpLite* op,
                                       KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NPU] Converting " + op_type + "...";

  // Get input, output and op attributes
  auto x_name = op_info->Input("Input").front();
  auto x = scope->FindTensor(x_name);
  auto x_dims = x->dims();

  auto out_name = op_info->Output("Out").front();
  auto out = scope->FindTensor(out_name);
  auto out_shape = out->dims().Vectorize();

  auto value = op_info->GetAttr<float>("value");

  // dims, value node
  std::vector<int> target_shape{out_shape.begin(), out_shape.end()};
  auto dims_node = graph->Add(out_name + "/dims", target_shape);

  auto value_node = graph->Add(out_name + "/value", std::vector<float>{value});

  // Fill node
  auto fill_node = graph->Add<ge::op::Fill>(out_name);
  auto fill_op = fill_node->data<ge::op::Fill>();
  fill_op->set_input_dims(*dims_node->data());
  fill_op->set_input_value(*value_node->data());

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    fill_constant_batch_size_like,
    kNPU,
    paddle::lite::subgraph::npu::FillConstantBatchSizeLikeConverter);
