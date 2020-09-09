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

#include "lite/operators/reshape_op.h"
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/huawei_ascend_npu/bridges/graph.h"
#include "lite/kernels/huawei_ascend_npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace huawei_ascend_npu {

int ReshapeConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Converting " + op_type + "...";

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

  // Shape Const node
  if (op_info->HasInput("ShapeTensor")) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] not support \"Shape\" from more than "
                    "one Tensor.";
    return FAILED;
  }

  std::shared_ptr<Node> actual_shape_node = nullptr;
  if (op_info->HasInput("Shape")) {
    auto actual_shape_name = op_info->Input("Shape").front();
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
      actual_shape_node =
          graph->Add<int>(actual_shape_name,
                          std::vector<int>(out_shape.begin(), out_shape.end()));
    }
  } else if (op_info->HasAttr("shape")) {
    auto shape = op_info->GetAttr<std::vector<int>>("shape");
    auto out_shape = lite::operators::ValidateShape(shape, x_dims);
    out_shape = CvtShape(out_shape);
    actual_shape_node = graph->Add<int64_t>(
        out_name + "/shape",
        std::vector<int64_t>(out_shape.begin(), out_shape.end()));
  }
  // actual_shape_node should not be nullptr
  CHECK(actual_shape_node);

  // Reshape node
  auto reshape_node = graph->Add<ge::op::Reshape>(out_name);
  auto reshape_op = reshape_node->data<ge::op::Reshape>();
  reshape_op->set_input_x(*x_node->data());
  reshape_op->set_input_shape(*actual_shape_node->data());
  INPUT_UPDATE(reshape_op, x, x_node);
  INPUT_UPDATE(reshape_op, shape, actual_shape_node);
  OUTPUT_UPDATE(reshape_op, y, reshape_node);

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace huawei_ascend_npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    reshape,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::ReshapeConverter);
REGISTER_SUBGRAPH_BRIDGE(
    reshape2,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::ReshapeConverter);
