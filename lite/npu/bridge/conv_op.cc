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

#include "lite/operators/conv_op.h"
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

node_map_type ConvConverter(const std::shared_ptr<lite::OpLite> conv_op,
                            const node_map_type& inputs_map) {
  lite::Scope* scope = conv_op->scope();
  const lite::OpInfo* op_info = conv_op->op_info();
  // build conv op node
  std::shared_ptr<ge::op::Convolution> output_node =
      std::make_shared<ge::op::Convolution>(UniqueName("conv2d"));

  auto x_var_name = op_info->Input("Input").front();
  CHECK(inputs_map.count(x_var_name));
  output_node->set_input_x(*inputs_map.at(x_var_name));
  // TODO(xxx): all other node should be from inputs_map
  // build filter and bias node
  auto filter_var_name = op_info->Input("Filter").front();
  lite::Tensor* filter =
      scope->FindVar(filter_var_name)->GetMutable<lite::Tensor>();
  auto filter_dims = filter->dims();
  ge::op::Const filter_const_node =
      ge::op::Const(filter_var_name).set_attr_value(CvtFromLiteTensor(filter));
  output_node->set_input_w(filter_const_node);
  if (op_info->HasInput("Bias")) {
    auto bias_var_names = op_info->Input("Bias");
    if (bias_var_names.size() > 0) {
      auto bias_var_name = bias_var_names.front();
      lite::Tensor* bias =
          scope->FindVar(bias_var_name)->GetMutable<lite::Tensor>();
      ge::op::Const bias_const_node =
          ge::op::Const(bias_var_name).set_attr_value(CvtFromLiteTensor(bias));
      output_node->set_input_b(bias_const_node);
    }
  }
  // set attributes
  std::vector<int> strides = op_info->GetAttr<std::vector<int>>("strides");
  std::vector<int> paddings = op_info->GetAttr<std::vector<int>>("paddings");
  int groups = op_info->GetAttr<int>("groups");
  std::vector<int> dilations = op_info->GetAttr<std::vector<int>>("dilations");
  output_node->set_attr_pad_mode(0);  // NOTSET
  output_node->set_attr_group(groups);
  output_node->set_attr_pad(
      ge::AttrValue::LIST_INT(paddings.begin(), paddings.end()));
  output_node->set_attr_dilation(
      ge::AttrValue::LIST_INT(dilations.begin(), dilations.end()));
  output_node->set_attr_stride(
      ge::AttrValue::LIST_INT(strides.begin(), strides.end()));
  output_node->set_attr_kernel(
      ge::AttrValue::LIST_INT({filter_dims[2], filter_dims[3]}));
  // get input&output tensor info
  auto input_var_name = op_info->Input("Input").front();
  lite::Tensor* input =
      scope->FindVar(input_var_name)->GetMutable<lite::Tensor>();
  int ic = input->dims()[1];
  auto output_var_name = op_info->Output("Output").front();
  lite::Tensor* output =
      scope->FindVar(output_var_name)->GetMutable<lite::Tensor>();
  int oc = output->dims()[1];
  // set depthwise mode if input_channel_num = output_channel_num = groups
  if (ic == oc && ic == groups) {
    output_node->set_attr_mode(3);  // depthwise_conv2d
  } else {
    output_node->set_attr_mode(1);
  }

  node_map_type outputs_map;
  outputs_map[op_info->Output("Output").front()] = output_node;
  return outputs_map;
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(conv2d, paddle::lite::npu::bridge::ConvConverter);
REGISTER_NPU_BRIDGE(depthwise_conv2d, paddle::lite::npu::bridge::ConvConverter);
