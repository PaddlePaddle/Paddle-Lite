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

#include "lite/operators/fc_op.h"
#include "ai_ddk_lib/include/graph/buffer.h"
#include "ai_ddk_lib/include/graph/graph.h"
#include "ai_ddk_lib/include/graph/model.h"
#include "ai_ddk_lib/include/graph/op/all_ops.h"
#include "ai_ddk_lib/include/graph/operator.h"
#include "ai_ddk_lib/include/graph/operator_reg.h"
#include "lite/npu/bridge/registry.h"
#include "lite/npu/bridge/utils.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

node_map_type FCConverter(const std::shared_ptr<lite::OpLite> fc_op,
                          const node_map_type& inputs_map) {
  lite::Scope* scope = fc_op->scope();
  auto* op_info = fc_op->op_info();

  std::shared_ptr<ge::op::FullConnection> output_node =
      std::make_shared<ge::op::FullConnection>(UniqueName("fc"));
  auto x_var_name = op_info->Input("Input").front();
  auto w_var_name = op_info->Input("W").front();
  CHECK(inputs_map.count(x_var_name));
  CHECK(inputs_map.count(w_var_name));
  output_node->set_input_x(*inputs_map.at(x_var_name));
  output_node->set_input_w(*inputs_map.at(w_var_name));

  // build and set weight and bias node
  int in_num_col_dims =
      op_info->GetAttr<int>("in_num_col_dims");  // TODO(hong19860320)
  if (op_info->HasInput("Bias")) {
    auto bias_var_names = op_info->Input("Bias");
    if (bias_var_names.size() > 0) {
      auto bias_var_name = bias_var_names.front();
      lite::Tensor* bias =
          scope->FindVar(bias_var_name)->GetMutable<lite::Tensor>();
      ge::op::Const bias_const_node =
          ge::op::Const(bias_var_name).set_attr_value(CvtFromLiteTensor(bias));
      output_node->set_input_b(bias_const_node);
    }
  }

  node_map_type outputs_map;
  outputs_map[op_info->Output("Out").front()] = output_node;
  return outputs_map;
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(fc, paddle::lite::npu::bridge::FCConverter);
