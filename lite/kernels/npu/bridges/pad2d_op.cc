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

#include "lite/kernels/npu/bridges/graph.h"
#include "lite/kernels/npu/bridges/registry.h"
#include "lite/kernels/npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace npu {

int Pad2dConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NPU] Converting " + op_type + "...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x_type = kernel->GetInputDeclType("Input");
  CHECK(x_type->precision() == PRECISION(kFloat));
  CHECK(x_type->layout() == DATALAYOUT(kNCHW));
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto out_name = op_info->Output("Out").front();
  auto out_type = kernel->GetOutputDeclType("Out");
  CHECK(out_type->precision() == PRECISION(kFloat));
  CHECK(out_type->layout() == DATALAYOUT(kNCHW));
  auto padding = op_info->GetAttr<std::vector<int>>("paddings");
  CHECK_EQ(padding.size(), 4);

  // X node
  std::shared_ptr<ge::Operator> x_node = nullptr;
  if (graph->HasNode(x_name)) {
    x_node = graph->GetNode(x_name);
  } else {
    x_node = graph->AddNode(x_name, x_dims);
  }

  // Padding node
  int xds = x_dims.size();
  padding.insert(padding.begin(), xds * 2 - 4, 0);
  auto padding_const_node =
      graph->AddNode(out_name + "/padding", padding, {xds, 2});

  // Pad node
  auto pad2d_node = graph->AddNode<ge::op::Pad>(out_name);
  pad2d_node->set_input_x(*x_node);
  pad2d_node->set_input_padding(*padding_const_node);
  auto mode = op_info->GetAttr<std::string>("mode");
  if (mode == "constant") {
    // Pad value node
    auto pad_value = op_info->GetAttr<float>("pad_value");
    auto pad_value_const_node =
        graph->AddNode(out_name + "/pad_value", pad_value);
    pad2d_node->set_input_constant_values(*pad_value_const_node);
    pad2d_node->set_attr_T(0);  // type of pad_value:  0:float  3:int32
    pad2d_node->set_attr_mode(0);
  } else if (mode == "reflect") {
    LOG(WARNING) << "[NPU] pad mode " << mode << " isn't supported in HiAI DDK";
    pad2d_node->set_attr_mode(1);
    return FAILED;
  } else {
    LOG(WARNING) << "[NPU] pad mode " << mode << " isn't supported in HiAI DDK";
    return FAILED;
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
