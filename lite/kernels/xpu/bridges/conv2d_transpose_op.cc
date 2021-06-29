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

int Conv2dTransposeConverter(void* ctx, OpLite* op, KernelBase* kernel) {
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
  auto filter_name = op_info->Input("Filter").front();
  auto filter = scope->FindMutableTensor(filter_name);
  auto filter_dims = filter->dims();
  auto out_name = op_info->Output("Out").front();

  auto strides = op_info->GetAttr<std::vector<int>>("strides");
  std::vector<int> paddings = op_info->GetAttr<std::vector<int>>("paddings");
  auto groups = op_info->GetAttr<int>("groups");
  std::vector<int> dilations = op_info->GetAttr<std::vector<int>>("dilations");

  // Input node
  std::shared_ptr<Node> input_node = nullptr;
  if (graph->Has(input_name)) {
    input_node = graph->Get(input_name);
  } else {
    input_node = graph->Add(input_name, *input);
  }

  // Filter node
  auto filter_node = graph->Add(filter_name, *filter);

  // Conv node
  auto conv2d_transpose_attrs = xtcl::make_object<xtcl::network::Conv2DTransposeAttrs>();
  conv2d_transpose_attrs->strides = std::move(CvtShape<xtcl::xIndexExpr>(strides));
  conv2d_transpose_attrs->padding = std::move(CvtShape<xtcl::xIndexExpr>(paddings));
  conv2d_transpose_attrs->dilation = std::move(CvtShape<xtcl::xIndexExpr>(dilations));
  conv2d_transpose_attrs->groups = groups;
  // conv_attrs->channels = nullptr;
  conv2d_transpose_attrs->kernel_size = std::move(xtcl::Array<xtcl::xIndexExpr>(nullptr));
  conv2d_transpose_attrs->data_layout = "NCHW";
  conv2d_transpose_attrs->kernel_layout = "OIHW";
  conv2d_transpose_attrs->out_layout = "";
  // conv_attrs->out_dtype = "";

  graph->Add(out_name,
             graph->builder_.CreateConv2DTranspose(
                 *input_node->data(), *filter_node->data(), conv2d_transpose_attrs));
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace xpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(conv2d_transpose,
                         kXPU,
                         paddle::lite::subgraph::xpu::Conv2dTransposeConverter);
