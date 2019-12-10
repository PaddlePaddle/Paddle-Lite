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

node_map_type ArgmaxConverter(const std::shared_ptr<lite::OpLite> argmax_op,
                              const node_map_type& inputs_map) {
  auto scope = argmax_op->scope();
  auto op_info = argmax_op->op_info();
  auto op_type = op_info->Type();
  auto unique_op_type = lite::npu::UniqueName(op_type);
  LOG(INFO) << "[NPU] Converting " + op_type + "...";

  int axis = op_info->GetAttr<int64_t>("axis");

  std::shared_ptr<ge::op::ArgMax> argmax_node =
      std::make_shared<ge::op::ArgMax>(unique_op_type);

  auto x_var_name = op_info->Input("X").front();

  CHECK(inputs_map.count(x_var_name));
  argmax_node->set_input_x1(*inputs_map.at(x_var_name));
  lite::npu::OpList::Global().add(inputs_map.at(x_var_name));
  lite::npu::OpList::Global().add(argmax_node);

  Tensor x2_t;
  x2_t.Resize(std::vector<int64_t>{1});
  auto x2_t_data = x2_t.mutable_data<int>();
  x2_t_data[0] = axis;

  auto x2 = std::make_shared<ge::op::Const>(unique_op_type + "/axis");
  x2->set_attr_value(lite::npu::CvtTensor(&x2_t));
  argmax_node->set_input_x2(*x2);
  lite::npu::OpList::Global().add(x2);

  //  argmax_node->set_attr_axis(axis);
  // argmax only support output_type==int32
  // argmax_node->set_attr_output_type(3);

  node_map_type outputs_map;
  outputs_map[op_info->Output("Out").front()] = argmax_node;
  return outputs_map;
}

}  // namespace bridges
}  // namespace npu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(arg_max,
                    paddle::lite::kernels::npu::bridges::ArgmaxConverter);
