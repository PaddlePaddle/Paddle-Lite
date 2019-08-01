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
#include "lite/npu/bridge/registry.h"
#include "lite/npu/bridge/utils.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

node_map_type ScaleConverter(const std::shared_ptr<lite::OpLite> scale_op,
                             const node_map_type& inputs_map) {
  lite::Scope* scope = scale_op->scope();
  const lite::OpInfo* op_info = scale_op->op_info();
  std::shared_ptr<ge::op::Scale> output_node =
      std::make_shared<ge::op::Scale>(UniqueName("scale"));

  auto x_var_name = op_info->Input("X").front();
  CHECK(inputs_map.count(x_var_name));
  output_node->set_input_x(*inputs_map.at(x_var_name));
  // set attributes
  float scale = op_info->GetAttr<float>("scale");
  float bias = op_info->GetAttr<float>("bias");
  bool bias_after_scale = op_info->GetAttr<bool>("bias_after_scale");
  if (!bias_after_scale) {
    bias *= scale;
  }
  if (fabs(bias) > 1e-6f) {
    // get input tensor shape
    auto input_var_name = op_info->Input("X").front();
    lite::Tensor* input =
        scope->FindVar(input_var_name)->GetMutable<lite::Tensor>();
    auto input_shape = input->dims().Vectorize();
    // create bias tensor and build constant node
    ge::op::Const bias_const_node =
        ge::op::Const(UniqueName("bias"))
            .set_attr_value(CreateTensorAndFillData(bias, input_shape));
    output_node->set_input_bias(bias_const_node);
    output_node->set_attr_has_bias_value(true);
  }
  output_node->set_attr_filler_type("constant");
  output_node->set_attr_filler_value(scale);

  node_map_type outputs_map;
  outputs_map[op_info->Output("Out").front()] = output_node;
  return outputs_map;
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(scale, paddle::lite::npu::bridge::ScaleConverter);
