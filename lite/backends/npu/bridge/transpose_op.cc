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

#include "lite/operators/transpose_op.h"
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

node_map_type TransposeConverter(
    const std::shared_ptr<lite::OpLite> transpose_op,
    const node_map_type& inputs_map) {
  auto scope = transpose_op->scope();
  auto op_info = transpose_op->op_info();
  auto op_type = op_info->Type();
  auto unique_op_type = UniqueName(op_type);
  LOG(INFO) << "Converting " + op_type + "...";

  std::shared_ptr<ge::op::Permute> transpose_node =
      std::make_shared<ge::op::Permute>(unique_op_type);
  auto x_var_name = op_info->Input("X").front();

  // paddlelite doesn't have this input
  // w must be set, but it does nothing
  auto w_var_name = unique_op_type + "/w";
  auto* w = scope->Var(w_var_name)->GetMutable<Tensor>();
  w->Resize({1});
  auto* w_data = w->mutable_data<float>();
  for (int i = 0; i < w->numel(); i++) {
    w_data[i] = 1.f;
  }
  auto npu_w = std::make_shared<ge::op::Const>(w_var_name);
  npu_w->set_attr_value(CvtFromLiteTensor(w));
  OpList::Global().add(npu_w);

  auto axis = op_info->GetAttr<std::vector<int>>("axis");
  auto npu_axis = ge::AttrValue::LIST_INT(axis.begin(), axis.end());

  CHECK(inputs_map.count(x_var_name));
  transpose_node->set_input_x(*inputs_map.at(x_var_name));
  transpose_node->set_input_w(*npu_w);
  transpose_node->set_attr_order(npu_axis);

  OpList::Global().add(inputs_map.at(x_var_name));
  OpList::Global().add(transpose_node);

  node_map_type outputs_map;
  outputs_map[op_info->Output("Out").front()] = transpose_node;
  return outputs_map;
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(transpose, paddle::lite::npu::bridge::TransposeConverter);
REGISTER_NPU_BRIDGE(transpose2, paddle::lite::npu::bridge::TransposeConverter);
