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
#include "lite/backends/npu/bridge/registry.h"
#include "lite/backends/npu/bridge/utils.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

node_map_type Pad2dConverter(const std::shared_ptr<lite::OpLite> pad2d_op,
                             const node_map_type& inputs_map) {
  auto scope = pad2d_op->scope();
  auto op_info = pad2d_op->op_info();
  auto op_type = op_info->Type();
  auto unique_op_type = UniqueName(op_type);
  LOG(INFO) << "Converting " + op_type + "...";

  std::shared_ptr<ge::op::Pad> pad2d_node =
      std::make_shared<ge::op::Pad>(unique_op_type);
  auto x_var_name = op_info->Input("X").front();
  pad2d_node->set_input_x(*inputs_map.at(x_var_name));
  OpList::Global().add(inputs_map.at(x_var_name));
  OpList::Global().add(pad2d_node);

  auto mode = op_info->GetAttr<std::string>("mode");
  if (mode == "constant") {
    pad2d_node->set_attr_mode(0);
  } else if (mode == "reflect") {
    LOG(FATAL) << "NPU doesn't support this pad mod: " << mode;
    pad2d_node->set_attr_mode(1);
  } else {
    LOG(FATAL) << "NPU doesn't support this pad mod: " << mode;
  }

  auto x_dims = scope->FindTensor(x_var_name)->dims();
  auto padding = op_info->GetAttr<std::vector<int>>("paddings");
  CHECK_EQ(padding.size(), 4);
  int xds = x_dims.size();
  padding.insert(padding.begin(), xds * 2 - 4, 0);
  auto npu_padding =
      std::make_shared<ge::op::Const>(unique_op_type + "/padding");
  npu_padding->set_attr_value(CreateTensorAndFillData<int>(padding, {xds, 2}));
  pad2d_node->set_input_padding(*npu_padding);
  OpList::Global().add(npu_padding);

  if (mode == "constant") {
    auto pad_value = op_info->GetAttr<float>("pad_value");
    auto npu_pad_value =
        std::make_shared<ge::op::Const>(unique_op_type + "/pad_value");
    npu_pad_value->set_attr_value(CreateTensorAndFillData<float>({pad_value}));
    pad2d_node->set_input_constant_values(*npu_pad_value);
    OpList::Global().add(npu_pad_value);

    pad2d_node->set_attr_T(0);  // type of pad_value:  0:float  3:int32
  }

  node_map_type outputs_map;
  outputs_map[op_info->Output("Out").front()] = pad2d_node;
  return outputs_map;
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(pad2d, paddle::lite::npu::bridge::Pad2dConverter);
