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

node_map_type SoftmaxConverter(const std::shared_ptr<lite::OpLite> softmax_op,
                               const node_map_type& inputs_map) {
  auto scope = softmax_op->scope();
  auto op_info = softmax_op->op_info();
  auto op_type = op_info->Type();
  auto unique_op_type = UniqueName(op_type);
  LOG(INFO) << "Converting " + op_type + "...";

  std::shared_ptr<ge::op::Softmax> softmax_node =
      std::make_shared<ge::op::Softmax>(unique_op_type);
  auto x_var_name = op_info->Input("X").front();

  auto x_dims = scope->FindVar(x_var_name)->GetMutable<Tensor>()->dims();
  auto axis = op_info->GetAttr<int>("axis");
  if (x_dims.size() > 3) {
    CHECK(!(axis == 2 && x_dims[3] > 1))
        << "unsupported npu softmax params: axis = " << axis
        << "  :x_w = " << x_dims[3];
  }

  CHECK(inputs_map.count(x_var_name));
  softmax_node->set_input_x(*inputs_map.at(x_var_name));
  softmax_node->set_attr_axis(axis);

  OpList::Global().add(inputs_map.at(x_var_name));
  OpList::Global().add(softmax_node);

  node_map_type outputs_map;
  outputs_map[op_info->Output("Out").front()] = softmax_node;
  return outputs_map;
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(softmax, paddle::lite::npu::bridge::SoftmaxConverter);
