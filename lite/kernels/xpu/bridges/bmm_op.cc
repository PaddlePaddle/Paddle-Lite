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
#include "lite/kernels/xpu/bridges/graph.h"
#include "lite/kernels/xpu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace xpu {

int BmmConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[XPU] Converting " + op_type + "...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();

  auto y_name = op_info->Input("Y").front();
  auto y = scope->FindMutableTensor(y_name);
  auto y_dims = y->dims();

  auto out_name = op_info->Output("Out").front();
  auto out = scope->FindMutableTensor(out_name);
  auto out_dims = out->dims();

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x);
  }

  // Y node
  std::shared_ptr<Node> y_node = nullptr;
  if (graph->Has(y_name)) {
    y_node = graph->Get(y_name);
  } else {
    y_node = graph->Add(y_name, *y);
  }

  // Matmul node
  if (x_dims.size() > 2 && y_dims.size() >= 2) {
    // x: [B, ..., M, K], y: [B, ..., K, N], out: [B, ..., M, N]
    // x: [B, M, K], y: [K, N], out: [B, M, N]
    // Reshape and transposed X node
    if (x_dims.size() != 3) {
      auto m = static_cast<int>(x_dims[x_dims.size() - 2]);
      auto k = static_cast<int>(x_dims[x_dims.size() - 1]);
      x_node = graph->Add(
          x_name + "/reshape",
          graph->builder_.CreateReshape(*x_node->data(), {-1, m, k}));
    }
    // Reshape and transposed Y node
    if (y_dims.size() != 3) {
      auto k = static_cast<int>(y_dims[y_dims.size() - 2]);
      auto n = static_cast<int>(y_dims[y_dims.size() - 1]);
      y_node = graph->Add(
          y_name + "/reshape",
          graph->builder_.CreateReshape(*y_node->data(), {-1, k, n}));
    }
    if (x_dims[x_dims.size() - 1] != y_dims[y_dims.size() - 1]) {
      if (x_dims[x_dims.size() - 1] == y_dims[y_dims.size() - 2]) {
        y_node = graph->Add(
            y_name + "/reshape/transpose",
            graph->builder_.CreateTranspose(*y_node->data(), {0, 2, 1}));
      } else if (x_dims[x_dims.size() - 2] == y_dims[y_dims.size() - 1]) {
        x_node = graph->Add(
            x_name + "/reshape/transpose",
            graph->builder_.CreateTranspose(*x_node->data(), {0, 2, 1}));
      } else if (x_dims[x_dims.size() - 2] == y_dims[y_dims.size() - 2]) {
        x_node = graph->Add(
            x_name + "/reshape/transpose",
            graph->builder_.CreateTranspose(*x_node->data(), {0, 2, 1}));
        y_node = graph->Add(
            y_name + "/reshape/transpose",
            graph->builder_.CreateTranspose(*y_node->data(), {0, 2, 1}));
      }
    }

    // Matmul node
    auto matmul_node = graph->Add(
        out_name,
        graph->builder_.CreateBatchMatmul(*x_node->data(), *y_node->data()));
    if (out_dims.size() != 3) {
      graph->Add(out_name,
                 graph->builder_.CreateReshape(
                     *matmul_node->data(), CvtShape<xtcl::Integer>(out_dims)));
    }
  } else {
    LOG(FATAL) << "[XPU] Not supported.";
    return FAILED;
  }
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace xpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(bmm, kXPU, paddle::lite::subgraph::xpu::BmmConverter);
