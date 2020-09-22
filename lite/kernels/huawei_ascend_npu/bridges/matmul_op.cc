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

#include <cmath>

#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/huawei_ascend_npu/bridges/graph.h"
#include "lite/kernels/huawei_ascend_npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace huawei_ascend_npu {

int MatMulConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Converting " + op_type + "...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x = scope->FindTensor(x_name);
  auto x_dims = x->dims();

  if (x_dims.size() < 2) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] Input dims should be equal or large "
                    "than 2 in Huawei Ascend NPU DDK.";
    return FAILED;
  }

  auto y_name = op_info->Input("Y").front();
  auto y = scope->FindTensor(y_name);
  auto y_dims = y->dims();

  if (y_dims.size() < 2) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] Input dims should be equal or large "
                    "than 2 in Huawei Ascend NPU DDK.";
    return FAILED;
  }

  if (x_dims.size() != y_dims.size()) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] dims size of input x1 and x2 must be "
                    "same in Huawei Ascend NPU DDK.";
    return FAILED;
  }

  auto out_name = op_info->Output("Out").front();
  auto out = scope->FindTensor(out_name);
  auto out_dims = out->dims();

  bool transpose_x = op_info->GetAttr<bool>("transpose_X");
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
    INPUT_UPDATE(matmul_op, x1, x_node);
    INPUT_UPDATE(matmul_op, x2, y_node);
    OUTPUT_UPDATE(matmul_op, y, matmul_node);
  } else {
    matmul_node = graph->Add<ge::op::BatchMatMul>(out_name);
    auto matmul_op = matmul_node->data<ge::op::BatchMatMul>();
    matmul_op->set_input_x1(*x_node->data());
    matmul_op->set_input_x2(*y_node->data());
    matmul_op->set_attr_adj_x1(transpose_x);
    matmul_op->set_attr_adj_x2(transpose_y);
    INPUT_UPDATE(matmul_op, x1, x_node);
    INPUT_UPDATE(matmul_op, x2, y_node);
    OUTPUT_UPDATE(matmul_op, y, matmul_node);
  }

  if (fabs(alpha - 1.f) > 1e-6f) {
    auto scale_node = graph->Add<ge::op::Muls>(out_name);
    auto scale_op = scale_node->data<ge::op::Muls>();
    scale_op->set_input_x(*matmul_node->data());
    scale_op->set_attr_value(alpha);
    INPUT_UPDATE(scale_op, x, matmul_node);
    OUTPUT_UPDATE(scale_op, y, scale_node);
  }

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace huawei_ascend_npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    matmul,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::MatMulConverter);
