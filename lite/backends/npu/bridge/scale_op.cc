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

#include "lite/operators/scale_op.h"
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

node_map_type ScaleConverter(const std::shared_ptr<lite::OpLite> scale_op,
                             const node_map_type& inputs_map) {
  auto scope = scale_op->scope();
  auto op_info = scale_op->op_info();
  auto op_type = op_info->Type();
  auto unique_op_type = UniqueName(op_type);
  LOG(INFO) << "Converting " + op_type + "...";

  // get input, output and op attributes
  auto x_var_name = op_info->Input("X").front();
  auto x = scope->FindVar(x_var_name)->GetMutable<lite::Tensor>();
  auto x_dims = x->dims().Vectorize();
  CHECK_GE(x_dims.size(), 2);
  std::vector<int64_t> scale_bias_shape = {x_dims[1]};
  float scale = op_info->GetAttr<float>("scale");
  float bias = op_info->GetAttr<float>("bias");
  bool bias_after_scale = op_info->GetAttr<bool>("bias_after_scale");
  if (!bias_after_scale) {
    bias *= scale;
  }

  // create scale node and set input node from inputs_map
  auto scale_node = std::make_shared<ge::op::Scale>(unique_op_type);
  CHECK(inputs_map.count(x_var_name));
  scale_node->set_input_x(*inputs_map.at(x_var_name));
  OpList::Global().add(inputs_map.at(x_var_name));
  OpList::Global().add(scale_node);

  // add filter node(fill with scale)
  auto filter_const_node =
      std::make_shared<ge::op::Const>(unique_op_type + "/filter");
  filter_const_node->set_attr_value(
      CreateTensorAndFillData(scale, scale_bias_shape));
  scale_node->set_input_filter(*filter_const_node);
  OpList::Global().add(filter_const_node);

  // add bias node(fill with bias)
  if (fabs(bias) > 1e-6f) {
    auto bias_const_node =
        std::make_shared<ge::op::Const>(unique_op_type + "/bias");
    bias_const_node->set_attr_value(
        CreateTensorAndFillData(bias, scale_bias_shape));
    scale_node->set_input_bias(*bias_const_node);
    scale_node->set_attr_has_bias_value(true);
    OpList::Global().add(bias_const_node);
  }

  scale_node->set_attr_axis(1);

  node_map_type outputs_map;
  outputs_map[op_info->Output("Out").front()] = scale_node;
  return outputs_map;
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(scale, paddle::lite::npu::bridge::ScaleConverter);
