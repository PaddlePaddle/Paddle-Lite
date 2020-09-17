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

int PoolConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[XPU] Converting " + op_type + "...";

  // Get input, and attributes
  auto x_name = op_info->Input("X").front();
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto out_name = op_info->Output("Out").front();
  auto pooling_type = op_info->GetAttr<std::string>("pooling_type");
  auto ceil_mode = op_info->GetAttr<bool>("ceil_mode");
  std::vector<int> paddings = op_info->GetAttr<std::vector<int>>("paddings");
  auto global_pooling = op_info->GetAttr<bool>("global_pooling");
  auto ksize = op_info->GetAttr<std::vector<int>>("ksize");
  auto strides = op_info->GetAttr<std::vector<int>>("strides");
  auto exclusive = op_info->GetAttr<bool>("exclusive");

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x);
  }

  // Pool node
  if (pooling_type == "max") {
    if (global_pooling) {
      graph->Add(out_name,
                 graph->builder_.CreateGlobalMaxPool2D(*x_node->data()));
    } else {
      graph->Add(
          out_name,
          graph->builder_.CreateMaxPool2D(*x_node->data(),
                                          CvtShape<xtcl::xIndexExpr>(ksize),
                                          CvtShape<xtcl::xIndexExpr>(strides),
                                          CvtShape<xtcl::xIndexExpr>(paddings),
                                          "NCHW",
                                          ceil_mode));
    }
  } else if (pooling_type == "avg") {
    if (global_pooling) {
      graph->Add(out_name,
                 graph->builder_.CreateGlobalAvgPool2D(*x_node->data()));
    } else {
      // !exclusive ---> count_include_pad
      graph->Add(
          out_name,
          graph->builder_.CreateAvgPool2D(*x_node->data(),
                                          CvtShape<xtcl::xIndexExpr>(ksize),
                                          CvtShape<xtcl::xIndexExpr>(strides),
                                          CvtShape<xtcl::xIndexExpr>(paddings),
                                          "NCHW",
                                          ceil_mode,
                                          !exclusive));
    }
  } else {
    LOG(WARNING) << "[XPU] Unsupported pooling type: " << pooling_type;
    return FAILED;
  }
  return SUCCESS;
}

}  // namespace xpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(pool2d,
                         kXPU,
                         paddle::lite::subgraph::xpu::PoolConverter);
