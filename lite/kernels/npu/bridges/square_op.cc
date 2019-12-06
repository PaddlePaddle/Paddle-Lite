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

node_map_type SquareConverter(const std::shared_ptr<lite::OpLite> square_op,
                              const node_map_type& inputs_map) {
  auto scope = square_op->scope();
  auto op_info = square_op->op_info();
  auto op_type = op_info->Type();
  auto unique_op_type = lite::npu::UniqueName(op_type);
  LOG(INFO) << "[NPU] Converting " + op_type + "...";

  std::shared_ptr<ge::op::Square> square_node =
      std::make_shared<ge::op::Square>(unique_op_type);

  auto x_var_name = op_info->Input("X").front();

  CHECK(inputs_map.count(x_var_name));
  square_node->set_input_x(*inputs_map.at(x_var_name));

  lite::npu::OpList::Global().add(inputs_map.at(x_var_name));
  lite::npu::OpList::Global().add(square_node);

  node_map_type outputs_map;
  outputs_map[op_info->Output("Out").front()] = square_node;
  return outputs_map;
}

}  // namespace bridges
}  // namespace npu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(square,
                    paddle::lite::kernels::npu::bridges::SquareConverter);
