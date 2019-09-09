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

#include "lite/operators/split_op.h"
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
node_map_type SplitConverter(const std::shared_ptr<lite::OpLite> split_op,
                             const node_map_type& inputs_map) {
  lite::Scope* scope = split_op->scope();
  const lite::OpInfo* op_info = split_op->op_info();
  auto op_type = op_info->Type();
  auto unique_op_type = UniqueName(op_type);
  LOG(INFO) << "Converting " << op_type << " ... ";

  auto x_var_name = op_info->Input("X").front();
  auto axis = op_info->GetAttr<int>("axis");
  auto num = op_info->GetAttr<int>("num");
  auto sections = op_info->GetAttr<std::vector<int>>("sections");
  int64_t sections_num = static_cast<int64_t>(sections.size());

  std::shared_ptr<ge::op::Split> output_node =
      std::make_shared<ge::op::Split>(unique_op_type);
  CHECK(inputs_map.count(x_var_name));
  output_node->set_input_x(*inputs_map.at(x_var_name));
  OpList::Global().add(inputs_map.at(x_var_name));

  output_node->set_attr_axis(static_cast<int64_t>(axis));
  if (num > 0) {
    output_node->set_attr_output_num(static_cast<int64_t>(num));
  } else {
    output_node->set_attr_output_num(sections_num);
    auto size_split = ge::AttrValue::LIST_INT(sections.begin(), sections.end());
    output_node->set_attr_size_split(size_split);
  }

  node_map_type outputs_map;
  auto out_var_names = op_info->Output("Out");
  output_node->create_dynamic_output_y(out_var_names.size());
  int index = 1;
  for (auto out_var_name : out_var_names) {
    auto const_node = std::make_shared<ge::op::Const>(
        unique_op_type + "/const_zero" + std::to_string(index));
    const_node->set_attr_value(CreateTensorAndFillData(0));
    OpList::Global().add(const_node);
    auto add_node = std::make_shared<ge::op::Add>(unique_op_type + "/add" +
                                                  std::to_string(index));
    add_node->set_input_x1(*output_node, "y" + std::to_string(index));
    add_node->set_input_x2(*const_node);
    outputs_map[out_var_name] = add_node;
    OpList::Global().add(add_node);
    index++;
  }

  OpList::Global().add(output_node);
  return outputs_map;
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(split, paddle::lite::npu::bridge::SplitConverter);
