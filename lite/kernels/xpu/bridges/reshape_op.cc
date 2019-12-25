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

int ReshapeConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto scope = op->scope();
  auto op_type = op_info->Type();
  VLOG(3) << "[XPU] Converting " + op_type + "...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x_type = kernel->GetInputDeclType("X");
  CHECK(x_type->precision() == PRECISION(kFloat));
  CHECK(x_type->layout() == DATALAYOUT(kNCHW));
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto out_name = op_info->Output("Out").front();
  auto out_type = kernel->GetOutputDeclType("Out");
  CHECK(out_type->precision() == PRECISION(kFloat));
  CHECK(out_type->layout() == DATALAYOUT(kNCHW));

  // X node
  std::shared_ptr<xtcl::xExpr> x_node = nullptr;
  if (graph->HasNode(x_name)) {
    x_node = graph->GetNode(x_name);
  } else {
    x_node = graph->AddNode(x_name, x_dims);
  }

  std::vector<int> shape;
  if (HasInputArg(op_info, scope, "ShapeTensor")) {
    auto shape_tensor_names = op_info->Input("ShapeTensor");
    // auto shape_tensor_type = kernel->GetInputDeclType("ShapeTensor");
    // CHECK(shape_tensor_type->precision() == PRECISION(kInt32));
    // CHECK(shape_tensor_type->layout() == DATALAYOUT(kNCHW));
    for (auto shape_tensor_name : shape_tensor_names) {
      auto shape_tensor = scope->FindMutableTensor(shape_tensor_name);
      auto shape_tensor_data = shape_tensor->mutable_data<int>();
      shape.emplace_back(shape_tensor_data[0]);
    }
    CHECK_GT(shape.size(), 0)
        << "[XPU] ShapeError: When `shape` in ReshapeOp is a list or tuple "
           "which contains Tensor, the shape's size can't be zero. "
           "But received shape's size is "
        << shape.size();
  } else if (HasInputArg(op_info, scope, "Shape")) {
    auto actual_shape_name = op_info->Input("Shape").front();
    // auto actual_shape_type = kernel->GetInputDeclType("Shape");
    // CHECK(actual_shape_type->precision() == PRECISION(kInt32));
    // CHECK(actual_shape_type->layout() == DATALAYOUT(kNCHW));
    auto actual_shape = scope->FindMutableTensor(actual_shape_name);
    auto actual_shape_dims = actual_shape->dims();
    auto actual_shape_data = actual_shape->mutable_data<int>();
    auto shape = std::vector<int>(
        actual_shape_data, actual_shape_data + actual_shape_dims.production());
  } else if (op_info->HasAttr("shape")) {
    shape = op_info->GetAttr<std::vector<int>>("shape");
  } else {
    LOG(WARNING) << "[XPU] No new shape for reshape op";
    return FAILED;
  }
  auto out_dims = operators::ValidateShape(shape, x_dims);

  // Reshape node
  graph->AddNode(out_name,
                 graph->builder_.CreateReshape(
                     *x_node, CvtShape<xtcl::Integer>(out_dims)));
  return REBUILD_WHEN_SHAPE_CHANGED;
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
