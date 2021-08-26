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

#include "lite/core/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/npu/bridges/graph.h"
#include "lite/kernels/npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace npu {

int ReduceMeanConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NPU] Converting " + op_type + "...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto out_name = op_info->Input("Out").front();
  auto keep_dim = op_info->GetAttr<bool>("keep_dim");
  std::vector<int> dim = op_info->GetAttr<std::vector<int>>("dim");
  CHECK(!dim.empty()) << "[NPU] \"dim\" of reduce_mean should not be empty.";
  for (size_t i = 0; i < dim.size(); i++) {
    if (dim[i] < 0) {
      dim[i] += x_dims.size();
    }
  }
  std::stable_sort(dim.begin(), dim.end());

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x);
  }

  // Using ReduceSum + Scale to implement ReduceMean

  // Dim node
  auto dim_node = graph->Add(out_name + "/dim", dim);

  // Reduce Sum node
  auto reduce_sum_node = graph->Add<ge::op::ReduceSum>(out_name + "/reducesum");
  auto reduce_sum_op = reduce_sum_node->data<ge::op::ReduceSum>();
  reduce_sum_op->set_input_x(*x_node->data());
  reduce_sum_op->set_input_w(*dim_node->data());
  reduce_sum_op->set_attr_keep_dims(keep_dim);

  // Scale node
  auto scale_node = graph->Add<ge::op::Scale>(out_name);
  auto scale_op = scale_node->data<ge::op::Scale>();
  scale_op->set_input_x(*reduce_sum_node->data());
  scale_op->set_attr_axis(1);

  // Add filter node(fill with scale)
  float scale = 1;
  for (size_t i = 0; i < dim.size(); i++) {
    scale /= x_dims[dim[i]];
  }
  std::vector<int64_t> scale_bias_shape = x_dims.Vectorize();
  if (keep_dim) {
    for (size_t i = 0; i < dim.size(); i++) {
      scale_bias_shape[dim[i]] = 1;
    }
  } else {
    const int64_t kDelFlag = -2;
    for (size_t i = 0; i < dim.size(); ++i) {
      scale_bias_shape[dim[i]] = kDelFlag;
    }
    scale_bias_shape.erase(
        remove(scale_bias_shape.begin(), scale_bias_shape.end(), kDelFlag),
        scale_bias_shape.end());
  }
  auto filter_node = graph->Add(out_name + "/filter", scale, scale_bias_shape);
  scale_op->set_input_filter(*filter_node->data());
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(reduce_mean,
                         kNPU,
                         paddle::lite::subgraph::npu::ReduceMeanConverter);
