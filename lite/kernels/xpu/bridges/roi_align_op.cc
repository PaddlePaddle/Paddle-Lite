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

#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/xpu/bridges/graph.h"
#include "lite/kernels/xpu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace xpu {
static constexpr int kROISize = 4;

int RoiConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[XPU] Converting " + op_type + "...";

  // Get input, output and op attributes
  auto x_name = op_info->Input("X").front();
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto ROIs_name = op_info->Input("ROIs").front();
  auto ROIs = scope->FindMutableTensor(ROIs_name);
  auto ROIs_dims = ROIs->dims();
  auto out_name = op_info->Output("Out").front();
  auto spatial_scale = op_info->GetAttr<float>("spatial_scale");
  auto pooled_height = op_info->GetAttr<int>("pooled_height");
  auto pooled_width = op_info->GetAttr<int>("pooled_width");
  auto sampling_ratio = op_info->GetAttr<int>("sampling_ratio");

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x);
  }

  // ROIs node
  std::shared_ptr<Node> ROIs_node = nullptr;
  if (graph->Has(ROIs_name)) {
    ROIs_node = graph->Get(ROIs_name);
  } else {
    ROIs_node = graph->Add(ROIs_name, *ROIs);
  }

  //std::vector<int> pooled_size;
  xtcl::Array<xtcl::xIndexExpr> pooled_size;
  pooled_size.push_back(pooled_height);
  pooled_size.push_back(pooled_width);

  graph->Add(out_name,
             graph->builder_.CreateROIAlign(
                 *x_node->data(), *ROIs_node->data(), 
                                      pooled_size,
                                      spatial_scale,
                                      sampling_ratio,
                                      "NCHW",
                                      "NCHW"));
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace xpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(roi_align,
                         kXPU,
                         paddle::lite::subgraph::xpu::RoiConverter);
