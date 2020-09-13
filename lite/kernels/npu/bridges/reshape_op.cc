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

#include "lite/operators/reshape_op.h"
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/npu/bridges/graph.h"
#include "lite/kernels/npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace npu {

int ReshapeConverter(void* ctx, OpLite* op, KernelBase* kernel) {
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

  auto out_name = op_info->Output("Out").front();

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x);
  }

  // Reshape node
  auto reshape_node = graph->Add<ge::op::Reshape>(
      out_name, x_node->precision(), x_node->layout());
  auto reshape_op = reshape_node->data<ge::op::Reshape>();
  reshape_op->set_input_tensor(*x_node->data());

  // Read shape from "ShapeTensor"(input), or "Shape"(input), or "shape"(attr)
  if (HasInputArg(op_info, scope, "ShapeTensor")) {
    LOG(WARNING) << "[NPU] not support \"Shape\" from more than one Tensor.";
    return FAILED;
  } else if (HasInputArg(op_info, scope, "Shape")) {
    auto actual_shape_name = op_info->Input("Shape").front();
    // auto actual_shape_type = kernel->GetInputDeclType("Shape");
    // CHECK(actual_shape_type->precision() == PRECISION(kInt32));
    // CHECK(actual_shape_type->layout() == DATALAYOUT(kNCHW));
    std::shared_ptr<Node> actual_shape_node = nullptr;
    if (graph->Has(actual_shape_name)) {
      actual_shape_node = graph->Get(actual_shape_name);
    } else {
      auto actual_shape = scope->FindMutableTensor(actual_shape_name);
      auto actual_shape_dims = actual_shape->dims();
      auto actual_shape_data = actual_shape->mutable_data<int>();
      auto shape =
          std::vector<int>(actual_shape_data,
                           actual_shape_data + actual_shape_dims.production());
      auto out_shape = lite::operators::ValidateShape(shape, x_dims);
      if (out_shape.size() > 4) {
        LOG(WARNING) << "[NPU] HiAI DDK only supports less than 4 dimensions, "
                        "but Shape has "
                     << out_shape.size();
        return FAILED;
      }
      actual_shape_node =
          graph->Add(actual_shape_name,
                     std::vector<int>(out_shape.begin(), out_shape.end()));
    }
    reshape_op->set_input_w(*actual_shape_node->data());
  } else {
    auto shape = op_info->GetAttr<std::vector<int>>("shape");
    auto out_shape = lite::operators::ValidateShape(shape, x_dims);
    out_shape = CvtShape(out_shape);
    if (out_shape.size() > 4) {
      LOG(WARNING) << "[NPU] HiAI DDK only supports less than 4 dimensions, "
                      "but shape has "
                   << out_shape.size();
      return FAILED;
    }
    reshape_op->set_attr_shape(
        ge::AttrValue::LIST_INT(out_shape.begin(), out_shape.end()));
  }

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(reshape,
                         kNPU,
                         paddle::lite::subgraph::npu::ReshapeConverter);
REGISTER_SUBGRAPH_BRIDGE(reshape2,
                         kNPU,
                         paddle::lite::subgraph::npu::ReshapeConverter);
