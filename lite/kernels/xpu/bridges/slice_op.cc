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
#include "lite/kernels/xpu/bridges/graph.h"
#include "lite/kernels/xpu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace xpu {

int SliceConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[XPU] Converting " + op_type + "...";

  // Get input, output and op attributes
  auto input_name = op_info->Input("Input").front();
  auto input = scope->FindMutableTensor(input_name);
  auto input_dims = input->dims();
  auto out_name = op_info->Output("Out").front();
  auto axes = op_info->GetAttr<std::vector<int>>("axes");
  auto starts = op_info->GetAttr<std::vector<int>>("starts");
  auto ends = op_info->GetAttr<std::vector<int>>("ends");

  // Input node
  std::shared_ptr<Node> input_node = nullptr;
  if (graph->Has(input_name)) {
    input_node = graph->Get(input_name);
  } else {
    input_node = graph->Add(input_name, *input);
  }

  // Calculate the begin and end of the slice in all of
  // dimensions and Create slice node as the output node
  xtcl::Array<xtcl::Integer> begin, end, strides;
  for (size_t i = 0; i < input_dims.size(); ++i) {
    auto it = std::find(axes.cbegin(), axes.cend(), i);
    if (it == axes.cend()) {
      // If not found, don't slice this axis
      int s = 0;
      int e = input_dims[i];
      begin.push_back(s);
      end.push_back(e);
      strides.push_back(1);
    } else {
      int offset = it - axes.cbegin();
      int s = starts[offset];
      int e = ends[offset];
      begin.push_back(s);
      end.push_back(e);
      strides.push_back(1);
    }
  }
  graph->Add(out_name,
             graph->builder_.CreateStridedSlice(
                 *input_node->data(), begin, end, strides));
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace xpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(slice,
                         kXPU,
                         paddle::lite::subgraph::xpu::SliceConverter);
