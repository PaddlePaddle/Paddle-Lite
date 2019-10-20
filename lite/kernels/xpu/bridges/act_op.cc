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

#include "lite/backends/xpu/builder.h"
#include "lite/kernels/xpu/bridges/registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {
namespace bridges {

node_map_type ActConverter(const std::shared_ptr<lite::OpLite> act_op,
                           const node_map_type& inputs_map) {
  auto op_info = act_op->op_info();
  auto op_type = op_info->Type();
  auto unique_op_type = lite::xpu::UniqueName(op_type);
  LOG(INFO) << "Converting " + op_type + "...";

  // check network context
  CHECK(inputs_map.network_builder != nullptr);
  CHECK(inputs_map.const_tensors != nullptr);

  // create activation node and set input nodes from inputs_map
  auto x_var_name = op_info->Input("X").front();
  CHECK(inputs_map.output_nodes.count(x_var_name));
  auto act_node =
      std::make_shared<xtcl::xExpr>(inputs_map.network_builder->CreateRelu(
          *inputs_map.output_nodes.at(x_var_name)));
  inputs_map.network_builder->SetLayer(unique_op_type);
  node_map_type outputs_map;
  outputs_map.network_builder = inputs_map.network_builder;
  outputs_map.const_tensors = inputs_map.const_tensors;
  outputs_map.output_nodes[op_info->Output("Out").front()] = act_node;
  return outputs_map;
}

}  // namespace bridges
}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_XPU_BRIDGE(relu, paddle::lite::kernels::xpu::bridges::ActConverter);
