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

node_map_type ElementwiseConverter(
    const std::shared_ptr<lite::OpLite> elementwise_op,
    const node_map_type& inputs_map) {
  auto scope = elementwise_op->scope();
  auto op_info = elementwise_op->op_info();
  auto op_type = op_info->Type();
  auto unique_op_type = lite::npu::UniqueName(op_type);
  LOG(INFO) << "[NPU] Converting " + op_type + "...";

  std::shared_ptr<ge::op::Eltwise> elementwise_node =
      std::make_shared<ge::op::Eltwise>(unique_op_type);

  auto x_var_name = op_info->Input("X").front();
  auto y_var_name = op_info->Input("Y").front();

  CHECK_EQ(op_info->GetAttr<int>("axis"), -1)
      << "[NPU] elementwise only support inputs with same size";

  CHECK(inputs_map.find(x_var_name) != inputs_map.end());
  elementwise_node->set_input_x1(*inputs_map.at(x_var_name));
  lite::npu::OpList::Global().add(inputs_map.at(x_var_name));

  if (inputs_map.find(y_var_name) != inputs_map.end()) {
    elementwise_node->set_input_x2(*inputs_map.at(y_var_name));
    lite::npu::OpList::Global().add(inputs_map.at(y_var_name));
  } else {
    auto y_const_node = std::make_shared<ge::op::Const>(y_var_name);
    auto* y = scope->FindVar(y_var_name)->GetMutable<Tensor>();
    y_const_node->set_attr_value(lite::npu::CvtTensor(y));
    elementwise_node->set_input_x2(*y_const_node);
    lite::npu::OpList::Global().add(y_const_node);
  }

  lite::npu::OpList::Global().add(elementwise_node);

  // paddlelite has sum only
  elementwise_node->set_attr_mode(1);

  node_map_type outputs_map;
  if (op_type == "fusion_elementwise_add_activation") {
    auto act_type = op_info->GetAttr<std::string>("act_type");
    auto act_node =
        std::make_shared<ge::op::Activation>(unique_op_type + "/act");
    act_node->set_input_x(*elementwise_node);
    // TODO(hong19860320) set the coef value for act Ops, such as leaky_relu,
    // clipped_relu etc.
    act_node->set_attr_mode(lite::npu::CvtActMode(act_type));
    lite::npu::OpList::Global().add(act_node);
    outputs_map[op_info->Output("Out").front()] = act_node;
  } else {
    outputs_map[op_info->Output("Out").front()] = elementwise_node;
  }
  return outputs_map;
}

}  // namespace bridges
}  // namespace npu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(elementwise_add,
                    paddle::lite::kernels::npu::bridges::ElementwiseConverter);
REGISTER_NPU_BRIDGE(fusion_elementwise_add_activation,
                    paddle::lite::kernels::npu::bridges::ElementwiseConverter);
