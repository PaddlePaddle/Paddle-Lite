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

#include "ai_ddk_lib/include/graph/buffer.h"
#include "ai_ddk_lib/include/graph/graph.h"
#include "ai_ddk_lib/include/graph/model.h"
#include "ai_ddk_lib/include/graph/op/all_ops.h"
#include "ai_ddk_lib/include/graph/operator.h"
#include "ai_ddk_lib/include/graph/operator_reg.h"
#include "lite/npu/bridge/registry.h"
#include "lite/npu/bridge/utils.h"
#include "lite/operators/relu_op.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

node_map_type ActConverter(const std::shared_ptr<lite::OpLite> act_op,
                           const node_map_type& inputs_map) {
  lite::Scope* scope = act_op->scope();
  const lite::OpInfo* op_info = act_op->op_info();
  const std::string act_type = op_info->Type();

  // build act op node
  std::shared_ptr<ge::op::Activation> output_node =
      std::make_shared<ge::op::Activation>(UniqueName("act"));
  auto x_var_name = op_info->Input("X").front();
  CHECK(inputs_map.count(x_var_name));
  output_node->set_input_x(*inputs_map.at(x_var_name));
  OpList::Global().add(inputs_map.at(x_var_name));
  OpList::Global().add(output_node);

  // set attributes
  int act_mode = 1;
  if (act_type == "sigmod") {
    act_mode = 0;
  } else if (act_type == "relu") {
    act_mode = 1;
  } else if (act_type == "tanh") {
    act_mode = 2;
  } else if (act_type == "elu") {
    act_mode = 4;
  } else if (act_type == "abs") {
    act_mode = 6;
  } else if (act_type == "softsign") {
    act_mode = 8;
  } else if (act_type == "softplus") {
    act_mode = 9;
  } else if (act_type == "hardsigmoid") {
    act_mode = 10;
  } else {
    // TODO(hong19860320) add more activation mode, and set the coef value
    // clipped ReLU, LEAKY_RELU, relu1, threshold, selu and linear
    LOG(FATAL) << "Unsupported activation type " << act_type;
  }
  output_node->set_attr_mode(act_mode);

  node_map_type outputs_map;
  outputs_map[op_info->Output("Out").front()] = output_node;

  return outputs_map;
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(sigmod, paddle::lite::npu::bridge::ActConverter);
REGISTER_NPU_BRIDGE(relu, paddle::lite::npu::bridge::ActConverter);
REGISTER_NPU_BRIDGE(tanh, paddle::lite::npu::bridge::ActConverter);
REGISTER_NPU_BRIDGE(elu, paddle::lite::npu::bridge::ActConverter);
REGISTER_NPU_BRIDGE(abs, paddle::lite::npu::bridge::ActConverter);
REGISTER_NPU_BRIDGE(softsign, paddle::lite::npu::bridge::ActConverter);
REGISTER_NPU_BRIDGE(softplus, paddle::lite::npu::bridge::ActConverter);
REGISTER_NPU_BRIDGE(hardsigmoid, paddle::lite::npu::bridge::ActConverter);
