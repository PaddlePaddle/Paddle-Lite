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

#include "lite/kernels/npu/bridges/graph.h"
#include "lite/kernels/npu/bridges/registry.h"
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
  auto x_type = kernel->GetInputDeclType("X");
  CHECK(x_type->precision() == PRECISION(kFloat));
  CHECK(x_type->layout() == DATALAYOUT(kNCHW));
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto out_name = op_info->Input("Out").front();
  auto out_type = kernel->GetOutputDeclType("Out");
  CHECK(out_type->precision() == PRECISION(kFloat));
  CHECK(out_type->layout() == DATALAYOUT(kNCHW));
  auto keep_dim = op_info->GetAttr<bool>("keep_dim");
  auto dim = op_info->GetAttr<std::vector<int>>("dim");
  CHECK(!dim.empty()) << "[NPU] \"dim\" of reduce_mean should not be empty.";
  for (size_t i = 0; i < dim.size(); i++) {
    if (dim[i] < 0) {
      dim[i] += x_dims.size();
    }
  }
  std::sort(dim.begin(), dim.end());

  // X node
  std::shared_ptr<ge::Operator> x_node = nullptr;
  if (graph->HasNode(x_name)) {
    x_node = graph->GetNode(x_name);
  } else {
    x_node = graph->AddNode(x_name, x_dims);
  }

  // Using ReduceSum + Scale to implement ReduceMean

  // Dim node
  auto dim_const_node = graph->AddNode(out_name + "/dim", dim);

  // Reduce Sum node
  auto reduce_sum_node =
      graph->AddNode<ge::op::ReduceSum>(out_name + "/reducesum");
  reduce_sum_node->set_input_x(*x_node);
  reduce_sum_node->set_input_w(*dim_const_node);
  reduce_sum_node->set_attr_keep_dims(keep_dim);

  // Scale node
  auto scale_node = graph->AddNode<ge::op::Scale>(out_name);
  scale_node->set_input_x(*reduce_sum_node);
  scale_node->set_attr_axis(1);

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
  auto filter_const_node =
      graph->AddNode(out_name + "/filter", scale, scale_bias_shape);
  scale_node->set_input_filter(*filter_const_node);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(NPU,
                         reduce_mean,
                         paddle::lite::subgraph::npu::ReduceMeanConverter);
