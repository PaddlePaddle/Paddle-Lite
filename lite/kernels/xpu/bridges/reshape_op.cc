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
#include "lite/kernels/npu/bridges/registry.h"
#include "lite/kernels/xpu/bridges/graph.h"
#include "lite/kernels/xpu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace xpu {

int ReshapeConverter(void* ctx, OpLite* op) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto scope = op->scope();
  auto op_type = op_info->Type();
  VLOG(3) << "[XPU] Converting " + op_type + "...";

  // Create node and set params from op
  auto x_var_name = op_info->Input("X").front();
  auto out_var_name = op_info->Output("Out").front();

  std::vector<int> shape;
  if (op_info->HasInput("ShapeTensor") &&
      !op_info->Input("ShapeTensor").empty()) {
    for (auto var_name : op_info->Input("ShapeTensor")) {
      shape.emplace_back(scope->FindMutableTensor(var_name)->data<int>()[0]);
    }
    CHECK_GT(shape.size(), 0)
        << "ShapeError: When `shape` in ReshapeOp is a list or tuple "
           "which contains Tensor, the shape's size can't be zero. "
           "But received shape's size is "
        << shape.size();
  } else if (op_info->HasInput("Shape") && !op_info->Input("Shape").empty()) {
    auto shape_tensor =
        scope->FindMutableTensor(op_info->Input("Shape").front());
    auto shape_data = shape_tensor->data<int>();
    shape = std::vector<int>(shape_data, shape_data + shape_tensor->numel());
  } else if (op_info->HasAttr("shape")) {
    shape = op_info->GetAttr<std::vector<int>>("shape");
  } else {
    LOG(FATAL) << "no new shape for reshape op";
  }
  auto out_dims =
      operators::ValidateShape(shape, scope->FindTensor(x_var_name)->dims());

  CHECK(graph->HasNode(x_var_name));
  graph->AddNode(out_var_name,
                 graph->builder_.CreateReshape(*graph->GetNode(x_var_name),
                                               Cvt2ArrayInt(out_dims)));

  return SUCCESS;
}

}  // namespace xpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(XPU,
                         reshape2,
                         paddle::lite::subgraph::xpu::ReshapeConverter);
REGISTER_SUBGRAPH_BRIDGE(XPU,
                         reshape,
                         paddle::lite::subgraph::xpu::ReshapeConverter);
