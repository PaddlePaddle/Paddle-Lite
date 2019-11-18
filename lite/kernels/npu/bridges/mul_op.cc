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

// Note: inputs_map the var_name contains only the data, the weight should be
// handle in this converter
node_map_type MulConverter(const std::shared_ptr<lite::OpLite> mul_op,
                           const node_map_type& inputs_map) {
  auto scope = mul_op->scope();
  auto op_info = mul_op->op_info();
  auto op_type = op_info->Type();
  auto unique_op_type = lite::npu::UniqueName(op_type);
  LOG(INFO) << "[NPU] Converting " + op_type + "...";

  auto x_var_name = op_info->Input("X").front();
  auto y_var_name = op_info->Input("Y").front();
  auto x = scope->FindVar(x_var_name)->GetMutable<lite::Tensor>();
  auto y = scope->FindVar(y_var_name)->GetMutable<lite::Tensor>();
  auto x_dims = x->dims();
  auto y_dims = y->dims();
  int x_num_col_dims = op_info->GetAttr<int>("x_num_col_dims");
  int y_num_col_dims = op_info->GetAttr<int>("y_num_col_dims");
  int m = x_dims.Slice(0, x_num_col_dims).production();
  int k = x_dims.Slice(x_num_col_dims, x_dims.size()).production();
  CHECK_EQ(k, y_dims.Slice(0, y_num_col_dims).production())
      << "[NPU] columns of X must be equal with rows of Y";
  int n = y_dims.Slice(y_num_col_dims, y_dims.size()).production();
  LOG(INFO) << "m:" << m << ",n:" << n << ",k:" << k;
  LOG(INFO) << "x_var_name:" << x_var_name
            << ", is data: " << inputs_map.count(x_var_name);
  LOG(INFO) << "y_var_name:" << y_var_name
            << ", is data: " << inputs_map.count(y_var_name);
  CHECK(inputs_map.count(x_var_name))
      << "[NPU] MatMul in HiAI DDK only support X is data, Y is const yet.";

  auto mul_node = std::make_shared<ge::op::MatMul>(unique_op_type);
  // add input x node which supports persistable and non-persistable tensor, and
  // reshape to (m, k)
  if (inputs_map.count(x_var_name)) {
    auto reshaped_x_node =
        std::make_shared<ge::op::Reshape>(x_var_name + "_reshape");
    reshaped_x_node->set_input_tensor(*inputs_map.at(x_var_name));
    reshaped_x_node->set_attr_shape({m, k});
    reshaped_x_node->set_attr_axis(0);
    mul_node->set_input_x1(*reshaped_x_node);
    lite::npu::OpList::Global().add(inputs_map.at(x_var_name));
    lite::npu::OpList::Global().add(reshaped_x_node);
  } else {
    auto x_const_node = std::make_shared<ge::op::Const>(x_var_name);
    x_const_node->set_attr_value(lite::npu::CvtTensor(x, {m, k}));
    mul_node->set_input_x1(*x_const_node);
    lite::npu::OpList::Global().add(x_const_node);
  }
  // add input y node which only supports persistable tensor, and reshape to (k,
  // n)
  if (inputs_map.count(y_var_name)) {
    auto reshaped_y_node =
        std::make_shared<ge::op::Reshape>(y_var_name + "_reshape");
    reshaped_y_node->set_input_tensor(*inputs_map.at(y_var_name));
    reshaped_y_node->set_attr_shape({k, n});
    reshaped_y_node->set_attr_axis(0);
    mul_node->set_input_x2(*reshaped_y_node);
    lite::npu::OpList::Global().add(inputs_map.at(y_var_name));
    lite::npu::OpList::Global().add(reshaped_y_node);
  } else {
    auto y_const_node = std::make_shared<ge::op::Const>(y_var_name);
    y_const_node->set_attr_value(lite::npu::CvtTensor(y, {k, n}));
    mul_node->set_input_x2(*y_const_node);
    lite::npu::OpList::Global().add(y_const_node);
  }

  lite::npu::OpList::Global().add(mul_node);

  node_map_type outputs_map;
  outputs_map[op_info->Output("Out").front()] = mul_node;
  return outputs_map;
}

}  // namespace bridges
}  // namespace npu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(mul, paddle::lite::kernels::npu::bridges::MulConverter);
