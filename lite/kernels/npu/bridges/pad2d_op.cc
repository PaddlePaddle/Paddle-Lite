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

#include "lite/core/mir/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/npu/bridges/context.h"
#include "lite/kernels/npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace npu {

int Pad2dConverter(void* ctx, OpLite* op) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph_ctx = static_cast<Context*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NPU] Converting " + op_type + "...";

  auto x_var_name = op_info->Input("X").front();
  auto out_var_name = op_info->Output("Out").front();
  auto pad2d_node = graph_ctx->AddNode<ge::op::Pad>(out_var_name);
  pad2d_node->set_input_x(*graph_ctx->GetNode(x_var_name));

  auto mode = op_info->GetAttr<std::string>("mode");
  if (mode == "constant") {
    pad2d_node->set_attr_mode(0);
  } else if (mode == "reflect") {
    LOG(WARNING) << "[NPU] pad mode " << mode << " isn't supported in HiAI DDK";
    pad2d_node->set_attr_mode(1);
    return FAILED;
  } else {
    LOG(WARNING) << "[NPU] pad mode " << mode << " isn't supported in HiAI DDK";
    return FAILED;
  }

  auto x_dims = scope->FindTensor(x_var_name)->dims();
  auto padding = op_info->GetAttr<std::vector<int>>("paddings");
  CHECK_EQ(padding.size(), 4);
  int xds = x_dims.size();
  padding.insert(padding.begin(), xds * 2 - 4, 0);
  auto padding_const_node =
      graph_ctx->AddNode(out_var_name + "/padding", padding, {xds, 2});
  pad2d_node->set_input_padding(*padding_const_node);

  if (mode == "constant") {
    auto pad_value = op_info->GetAttr<float>("pad_value");
    auto pad_value_const_node = graph_ctx->AddNode(
        out_var_name + "/pad_value", std::vector<float>({pad_value}));
    pad2d_node->set_input_constant_values(*pad_value_const_node);
    pad2d_node->set_attr_T(0);  // type of pad_value:  0:float  3:int32
  }
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(NPU,
                         pad2d,
                         paddle::lite::subgraph::npu::Pad2dConverter);
