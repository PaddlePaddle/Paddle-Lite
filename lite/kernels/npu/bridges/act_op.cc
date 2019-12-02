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

#include "lite/backends/npu/builder.h"
#include "lite/kernels/npu/bridges/registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace npu {
namespace bridges {

node_map_type ActConverter(const std::shared_ptr<lite::OpLite> act_op,
                           const node_map_type& inputs_map) {
  auto scope = act_op->scope();
  auto op_info = act_op->op_info();
  auto op_type = op_info->Type();
  auto unique_op_type = lite::npu::UniqueName(op_type);
  LOG(INFO) << "[NPU] Converting " + op_type + "...";

  // create act node and set input node from inputs_map
  auto x_var_name = op_info->Input("X").front();
  auto act_node = std::make_shared<ge::op::Activation>(unique_op_type);
  CHECK(inputs_map.count(x_var_name));
  act_node->set_input_x(*inputs_map.at(x_var_name));
  lite::npu::OpList::Global().add(inputs_map.at(x_var_name));
  lite::npu::OpList::Global().add(act_node);

  // TODO(hong19860320) set the coef value for act Ops, such as leaky_relu,
  // clipped_relu etc.
  act_node->set_attr_mode(lite::npu::CvtActMode(op_type));

  if (op_type == "relu_clipped") {
    auto Relu_clipped_coef = op_info->GetAttr<float>("Relu_clipped_coef");
    act_node->set_attr_coef(Relu_clipped_coef);
  } else if (op_type == "leaky_relu") {
    auto alpha = op_info->GetAttr<float>("alpha");
    act_node->set_attr_negative_slope(alpha);
  } else if (op_type == "hard_sigmoid") {
    auto slope = op_info->GetAttr<float>("slope");
    auto offset = op_info->GetAttr<float>("offset");
    act_node->set_attr_negative_slope(slope);
    act_node->set_attr_coef(offset);
  }

  node_map_type outputs_map;
  outputs_map[op_info->Output("Out").front()] = act_node;
  return outputs_map;
}

}  // namespace bridges
}  // namespace npu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(sigmoid, paddle::lite::kernels::npu::bridges::ActConverter);
REGISTER_NPU_BRIDGE(relu, paddle::lite::kernels::npu::bridges::ActConverter);
REGISTER_NPU_BRIDGE(tanh, paddle::lite::kernels::npu::bridges::ActConverter);
REGISTER_NPU_BRIDGE(relu_clipped,
                    paddle::lite::kernels::npu::bridges::ActConverter);
// REGISTER_NPU_BRIDGE(elu, paddle::lite::kernels::npu::bridges::ActConverter);
REGISTER_NPU_BRIDGE(leaky_relu,
                    paddle::lite::kernels::npu::bridges::ActConverter);
REGISTER_NPU_BRIDGE(abs, paddle::lite::kernels::npu::bridges::ActConverter);
REGISTER_NPU_BRIDGE(softsign,
                    paddle::lite::kernels::npu::bridges::ActConverter);
REGISTER_NPU_BRIDGE(softplus,
                    paddle::lite::kernels::npu::bridges::ActConverter);
REGISTER_NPU_BRIDGE(hard_sigmoid,
                    paddle::lite::kernels::npu::bridges::ActConverter);
