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

node_map_type ReduceMeanConverter(
    const std::shared_ptr<lite::OpLite> reduce_mean_op,
    const node_map_type& inputs_map) {
  auto scope = reduce_mean_op->scope();
  auto op_info = reduce_mean_op->op_info();
  auto op_type = op_info->Type();
  auto unique_op_type = lite::npu::UniqueName(op_type);
  LOG(INFO) << "[NPU] Converting " + op_type + "...";

  // get input, and op attributes
  auto x_var_name = op_info->Input("X").front();
  auto x_dims = scope->FindTensor(x_var_name)->dims();
  auto keep_dim = op_info->GetAttr<bool>("keep_dim");
  auto dim = op_info->GetAttr<std::vector<int>>("dim");
  CHECK(!dim.empty()) << "\"dim\" of reduce_mean should not be empty.";
  for (size_t i = 0; i < dim.size(); i++) {
    if (dim[i] < 0) {
      dim[i] += x_dims.size();
    }
  }
  std::sort(dim.begin(), dim.end());

  // create reduce_mean(reduce_sum + scale) node and set input node from
  // inputs_map
  // creat reduce_sum node
  auto unique_reduce_sum = lite::npu::UniqueName("reduce_sum");
  auto reduce_sum_node = std::make_shared<ge::op::ReduceSum>(unique_reduce_sum);
  CHECK(inputs_map.count(x_var_name));
  reduce_sum_node->set_input_x(*inputs_map.at(x_var_name));
  lite::npu::OpList::Global().add(inputs_map.at(x_var_name));
  lite::npu::OpList::Global().add(reduce_sum_node);

  auto dim_const_node =
      std::make_shared<ge::op::Const>(unique_reduce_sum + "/dim");
  dim_const_node->set_attr_value(lite::npu::CreateTensorAndFillData<int>(dim));
  reduce_sum_node->set_input_w(*dim_const_node);
  lite::npu::OpList::Global().add(dim_const_node);

  reduce_sum_node->set_attr_keep_dims(keep_dim);

  // create scale node
  auto unique_scale = lite::npu::UniqueName("scale");
  auto scale_node = std::make_shared<ge::op::Scale>(unique_scale);
  scale_node->set_input_x(*reduce_sum_node);
  lite::npu::OpList::Global().add(scale_node);

  float scale = 1;
  for (size_t i = 0; i < dim.size(); i++) {
    scale /= x_dims[dim[i]];
  }

  std::vector<int64_t> scale_bias_shape = x_dims.Vectorize();
  if (keep_dim) {
    for (size_t i = 0; i < dim.size(); i++) {
      scale_bias_shape[dim[i]] = 1;
    }
  } else {
    const int64_t kDelFlag = -2;
    for (size_t i = 0; i < dim.size(); ++i) {
      scale_bias_shape[dim[i]] = kDelFlag;
    }
    scale_bias_shape.erase(
        remove(scale_bias_shape.begin(), scale_bias_shape.end(), kDelFlag),
        scale_bias_shape.end());
  }

  auto filter_const_node =
      std::make_shared<ge::op::Const>(unique_scale + "/filter");
  filter_const_node->set_attr_value(
      lite::npu::CreateTensorAndFillData(scale, scale_bias_shape));
  scale_node->set_input_filter(*filter_const_node);
  lite::npu::OpList::Global().add(filter_const_node);

  scale_node->set_attr_axis(1);

  node_map_type outputs_map;
  outputs_map[op_info->Output("Out").front()] = scale_node;
  return outputs_map;
}

}  // namespace bridges
}  // namespace npu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(reduce_mean,
                    paddle::lite::kernels::npu::bridges::ReduceMeanConverter);
