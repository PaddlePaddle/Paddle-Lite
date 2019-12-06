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

node_map_type UnsqueezeConverter(
    const std::shared_ptr<lite::OpLite> unsqueeze_op,
    const node_map_type& inputs_map) {
  auto scope = unsqueeze_op->scope();
  auto op_info = unsqueeze_op->op_info();
  auto op_type = op_info->Type();
  auto unique_op_type = lite::npu::UniqueName(op_type);
  LOG(INFO) << "[NPU] Converting " + op_type + "...";

  std::shared_ptr<ge::op::Reshape> unsqueeze_node =
      std::make_shared<ge::op::Reshape>(unique_op_type);

  auto x_var_name = op_info->Input("X").front();
  CHECK(inputs_map.count(x_var_name));
  unsqueeze_node->set_input_tensor(*inputs_map.at(x_var_name));

  lite::npu::OpList::Global().add(inputs_map.at(x_var_name));
  lite::npu::OpList::Global().add(unsqueeze_node);

  CHECK(op_info->HasAttr("axes"))
      << "[NPU] unsqueeze not support axes from tensor now";
  auto out_var_name = op_info->Output("Out").front();
  auto out_shape = scope->FindTensor(out_var_name)->dims().Vectorize();
  unsqueeze_node->set_attr_shape(
      ge::AttrValue::LIST_INT(out_shape.begin(), out_shape.end()));

  node_map_type outputs_map;
  outputs_map[op_info->Output("Out").front()] = unsqueeze_node;
  return outputs_map;
}

}  // namespace bridges
}  // namespace npu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(unsqueeze,
                    paddle::lite::kernels::npu::bridges::UnsqueezeConverter);
REGISTER_NPU_BRIDGE(unsqueeze2,
                    paddle::lite::kernels::npu::bridges::UnsqueezeConverter);
