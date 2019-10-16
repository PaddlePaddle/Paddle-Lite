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

#include "lite/backends/xpu/bridge/registry.h"
#include "lite/backends/xpu/builder.h"

namespace paddle {
namespace lite {
namespace xpu {
namespace bridge {

node_map_type ActConverter(const std::shared_ptr<lite::OpLite> act_op,
                           const node_map_type& inputs_map) {
  // auto scope = act_op->scope();
  auto op_info = act_op->op_info();
  auto op_type = op_info->Type();
  auto unique_op_type = UniqueName(op_type);
  LOG(INFO) << "Converting " + op_type + "...";

  // create act node and set input node from inputs_map
  auto x_var_name = op_info->Input("X").front();
  auto act_node = std::make_shared<std::string>(unique_op_type);
  node_map_type outputs_map;
  outputs_map[op_info->Output("Out").front()] = act_node;
  return outputs_map;
}

}  // namespace bridge
}  // namespace xpu
}  // namespace lite
}  // namespace paddle

REGISTER_XPU_BRIDGE(relu, paddle::lite::xpu::bridge::ActConverter);
