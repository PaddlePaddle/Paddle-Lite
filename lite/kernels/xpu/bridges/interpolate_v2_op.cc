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

int BilinearInterpV2Converter(void* ctx, OpLite* op, KernelBase* kernel) {
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

  // x node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x);
  }

  auto out_name = op_info->Output("Out").front();

  auto align_corners = op_info->GetAttr<bool>("align_corners");
  auto data_layout = op_info->GetAttr<std::string>("data_layout");
  auto interp_method = op_info->GetAttr<std::string>("interp_method");

  CHECK(align_corners == true) << "Only align_corners==True is supported for "
                                  "PaddlePaddle's bilinear_interp_v2";
  CHECK(data_layout == "NCHW")
      << "Only NCHW is supported for PaddlePaddle's bilinear_interp_v2";
  CHECK(interp_method == "bilinear")
      << "Only bilinear is supported for PaddlePaddle's bilinear_interp_v2";

  if (op_info->GetAttr<int>("out_h") != -1 &&
      op_info->GetAttr<int>("out_w") != -1) {
    std::vector<tvm::relay::IndexExpr> resize;
    resize.push_back(op_info->GetAttr<int>("out_h"));
    resize.push_back(op_info->GetAttr<int>("out_w"));
    graph->Add(out_name,
               graph->builder_.CreateInterpolate(*x_node->data(),
                                                 resize,
                                                 data_layout,
                                                 interp_method,
                                                 align_corners));
  } else if (scope->FindMutableTensor(op_info->Input("OutSize").front()) !=
             nullptr) {
    CHECK(false) << "bilinear_interp_v2 kernel not support attr 'OutSize'.";
  } else if (op_info->GetAttr<std::vector<float>>("scale_v").size() > 0 ||
             op_info->GetAttr<float>("scale") != 0) {
    CHECK(false) << "bilinear_interp_v2 kernel not support attr 'scale'.";
  } else {
    CHECK(false) << "bilinear_interp_v2 kernel not support other size input.";
  }

  return SUCCESS;
}

}  // namespace xpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    bilinear_interp_v2,
    kXPU,
    paddle::lite::subgraph::xpu::BilinearInterpV2Converter);
