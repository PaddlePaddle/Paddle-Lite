// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

int BoxClipConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[XPU] Converting " + op_type + "...";

  // Get input and output vars and op attributes
  auto input_name = op_info->Input("Input").front();
  auto input = scope->FindMutableTensor(input_name);

  auto im_info_name = op_info->Input("ImInfo").front();
  auto im_info = scope->FindMutableTensor(im_info_name);

  auto output_name = op_info->Output("Output").front();

  // Input node
  std::shared_ptr<Node> input_node = nullptr;
  if (graph->Has(input_name)) {
    input_node = graph->Get(input_name);
  } else {
    input_node = graph->Add(input_name, *input);
  }

  // ImInfo node
  std::shared_ptr<Node> im_info_node = nullptr;
  if (graph->Has(im_info_name)) {
    im_info_node = graph->Get(im_info_name);
  } else {
    im_info_node = graph->Add(im_info_name, *im_info);
  }

  // Output node
  graph->Add(output_name,
             graph->builder_.CreateBoxClip(*input_node->data(),
                                           *im_info_node->data()));

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace xpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(box_clip,
                         kXPU,
                         paddle::lite::subgraph::xpu::BoxClipConverter);
