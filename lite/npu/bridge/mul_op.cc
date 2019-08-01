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

#include "lite/operators/mul_op.h"
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

node_map_type MulConverter(const std::shared_ptr<lite::OpLite> mul_op,
                           const node_map_type& inputs_map) {
  LOG(INFO) << "converting mul...";
  lite::Scope* scope = mul_op->scope();
  const lite::OpInfo* op_info = mul_op->op_info();
  // build mul op node
  std::shared_ptr<ge::op::Mul> output_node =
      std::make_shared<ge::op::Mul>(UniqueName("mul"));
  // set x and y node
  //   int x_num_col_dims =
  //       op_info->GetAttr<int>("x_num_col_dims");  // TODO(hong19860320)
  //   int y_num_col_dims =
  //       op_info->GetAttr<int>("y_num_col_dims");  // TODO(hong19860320)

  auto x_var_name = op_info->Input("X").front();
  auto y_var_name = op_info->Input("Y").front();
  LOG(INFO) << "x_var_name:" << x_var_name;
  LOG(INFO) << "y_var_name:" << y_var_name;
  CHECK(inputs_map.count(x_var_name));
  CHECK(inputs_map.count(y_var_name));
  LOG(INFO) << inputs_map.at(x_var_name);
  LOG(INFO) << inputs_map.at(y_var_name);
  output_node->set_input_x(*inputs_map.at(x_var_name));
  output_node->set_input_y(*inputs_map.at(y_var_name));

  node_map_type outputs_map;
  outputs_map[op_info->Output("Out").front()] = output_node;
  return outputs_map;
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(mul, paddle::lite::npu::bridge::MulConverter);
