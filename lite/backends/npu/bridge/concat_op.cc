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

#include "lite/operators/concat_op.h"
#include "ai_ddk_lib/include/graph/buffer.h"
#include "ai_ddk_lib/include/graph/graph.h"
#include "ai_ddk_lib/include/graph/model.h"
#include "ai_ddk_lib/include/graph/op/all_ops.h"
#include "ai_ddk_lib/include/graph/operator.h"
#include "ai_ddk_lib/include/graph/operator_reg.h"
#include "lite/backends/npu/bridge/registry.h"
#include "lite/backends/npu/bridge/utils.h"
#include "lite/backends/npu/npu_helper.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

node_map_type ConcatConverter(const std::shared_ptr<lite::OpLite> concat_op,
                              const node_map_type& inputs_map) {
  lite::Scope* scope = concat_op->scope();
  const lite::OpInfo* op_info = concat_op->op_info();
  auto op_type = op_info->Type();
  auto unique_op_type = UniqueName(op_type);
  LOG(INFO) << "converting " << op_type << " ... ";

  auto x_var_names = op_info->Input("X");
  auto axis = op_info->GetAttr<int>("axis");
  int num = x_var_names.size();
  int index = 0;

  std::shared_ptr<ge::op::Concat> output_node =
      std::make_shared<ge::op::Concat>(unique_op_type);
  output_node->set_attr_axis(axis);
  output_node->set_attr_N(num);
  output_node->create_dynamic_input_x(num);
  for (auto x_var_name : x_var_names) {
    if (inputs_map.find(x_var_name) != inputs_map.end()) {
      output_node->set_dynamic_input_x(index + 1, *inputs_map.at(x_var_name));
      OpList::Global().add(inputs_map.at(x_var_name));
    } else {
      auto consty = std::make_shared<ge::op::Const>(x_var_name);
      auto* x = scope->FindVar(x_var_name)->GetMutable<Tensor>();
      consty->set_attr_value(CvtFromLiteTensor(x));
      output_node->set_dynamic_input_x(index + 1, *consty);
      OpList::Global().add(consty);
    }
    index++;
  }
  OpList::Global().add(output_node);

  node_map_type outputs_map;
  outputs_map[op_info->Output("Out").front()] = output_node;
  return outputs_map;
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(concat, paddle::lite::npu::bridge::ConcatConverter);
