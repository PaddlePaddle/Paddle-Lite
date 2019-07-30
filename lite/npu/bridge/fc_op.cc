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

std::vector<std::shared_ptr<ge::Operator>> FCConverter(
    const std::shared_ptr<lite::OpLite> op,
    const std::vector<std::shared_ptr<ge::Operator>>& input_nodes) {
  const std::shared_ptr<lite::operators::FcOpLite> fc_op =
      static_pointer_cast<lite::operators::FcOpLite>(op);
  lite::Scope* scope = fc_op->scope();
  // build fc op node
  std::shared_ptr<ge::op::FullConnection> output_node =
      std::make_shared<ge::op::FullConnection>(UniqueName("fc"));
  output_node->set_input_x(*input_nodes[0]);
  // build and set weight and bias node
  const lite::OpInfo* op_info = fc_op->op_info();
  int in_num_col_dims =
      op_info->GetAttr<int>("in_num_col_dims");  // TODO(hong19860320)
  auto w_var_name = op_info->Input("W").front();
  lite::Tensor* w = scope->FindVar(w_var_name)->GetMutable<lite::Tensor>();
  ge::op::Const w_const_node =
      ge::op::Const(w_var_name).set_attr_value(TensorConverter(w));
  output_node->set_input_w(w_const_node);
  if (op_info->HasInput("Bias")) {
    auto bias_var_names = op_info->Input("Bias");
    if (bias_var_names.size() > 0) {
      auto bias_var_name = bias_var_names.front();
      lite::Tensor* bias =
          scope->FindVar(bias_var_name)->GetMutable<lite::Tensor>();
      ge::op::Const bias_const_node =
          ge::op::Const(bias_var_name).set_attr_value(TensorConverter(bias));
      output_node->set_input_b(bias_const_node);
    }
  }
  std::vector<std::shared_ptr<ge::Operator>> output_nodes;
  output_nodes.push_back(output_node);
  return output_nodes;
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(fc, paddle::lite::npu::bridge::FCConverter);
