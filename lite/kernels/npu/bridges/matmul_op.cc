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

#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/npu/bridges/graph.h"
#include "lite/kernels/npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace npu {

int MatMulConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NPU] Converting " + op_type + "...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x = scope->FindTensor(x_name);
  auto x_dims = x->dims();

  auto y_name = op_info->Input("Y").front();
  auto y = scope->FindTensor(y_name);
  auto y_dims = y->dims();

  if (x_dims.size() == 1 || x_dims.size() != y_dims.size()) {
    LOG(WARNING)
        << "[NPU] dims size of x and y must be same and greater than 1.";
    return FAILED;
  }
  if (y_dims.size() == 2 && !y->persistable()) {
    LOG(WARNING) << "[NPU] y must be const if y is 2-D";
    return FAILED;
  }
  if (x_dims.size() > 2 &&
      x_dims.count(0, x_dims.size() - 2) !=
          y_dims.count(0, y_dims.size() - 2)) {
    LOG(WARNING) << "[NPU] batched matmul only support the same batch size";
    return FAILED;
  }

  auto out_name = op_info->Output("Out").front();
  auto out = scope->FindTensor(out_name);
  auto out_dims = out->dims();

  bool transpose_x = op_info->GetAttr<bool>("transpose_X");
  if (x_dims.size() > 2 && transpose_x) {
    LOG(WARNING) << "[NPU] not support transpose_x == true if x_dims size "
                    "greater than 2.";
    return FAILED;
  }
  bool transpose_y = op_info->GetAttr<bool>("transpose_Y");
  float alpha = op_info->GetAttr<float>("alpha");

  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x);
  }

  std::shared_ptr<Node> y_node = nullptr;
  if (graph->Has(y_name)) {
    y_node = graph->Get(y_name);
  } else {
    y_node = graph->Add(y_name, *y);
  }

  // Matmul node
  std::shared_ptr<Node> matmul_node = nullptr;
  if (x_dims.size() == 2) {
    matmul_node = graph->Add<ge::op::MatMul>(out_name);
    auto matmul_op = matmul_node->data<ge::op::MatMul>();
    matmul_op->set_input_x1(*x_node->data());
    matmul_op->set_input_x2(*y_node->data());
    matmul_op->set_attr_transpose_x1(transpose_x);
    matmul_op->set_attr_transpose_x2(transpose_y);
  } else {
    matmul_node = graph->Add<ge::op::BatchMatMul>(out_name);
    auto matmul_op = matmul_node->data<ge::op::BatchMatMul>();
    matmul_op->set_input_x1(*x_node->data());
    matmul_op->set_input_x2(*y_node->data());
    matmul_op->set_attr_adj_x1(transpose_x);
    matmul_op->set_attr_adj_x2(transpose_y);
  }

  if (fabs(alpha - 1.f) > 1e-6f) {
    auto scaled_out_node = graph->Add<ge::op::Scale>(out_name);
    auto scaled_out_op = scaled_out_node->data<ge::op::Scale>();
    scaled_out_op->set_input_x(*matmul_node->data());
    scaled_out_op->set_attr_axis(1);
    std::vector<int64_t> scale_bias_shape(4, 1);
    if (out_dims.size() < 4) {
      scale_bias_shape[1] = out_dims[0];
    } else if (out_dims.size() == 4) {
      scale_bias_shape[1] = out_dims[1];
    } else {
      LOG(WARNING) << "[NPU] not support out dims size greater than 4.";
      return FAILED;
    }
    auto filter_node =
        graph->Add(out_name + "/filter", alpha, scale_bias_shape);
    scaled_out_op->set_input_filter(*filter_node->data());
  }

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(matmul,
                         kNPU,
                         paddle::lite::subgraph::npu::MatMulConverter);
