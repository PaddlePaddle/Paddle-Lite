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
  VLOG(3) << "invoking ConvConverter...";
  auto scope = conv_op->scope();
  auto op_info = conv_op->op_info();
  auto op_type = op_info->Type();

  // get input, output and op attributes
  auto input_var_name = op_info->Input("Input").front();
  auto input = scope->FindVar(input_var_name)->GetMutable<lite::Tensor>();
  auto input_dims = input->dims();
  auto filter_var_name = op_info->Input("Filter").front();
  auto filter = scope->FindVar(filter_var_name)->GetMutable<lite::Tensor>();
  auto filter_dims = filter->dims();
  CHECK_EQ(filter_dims.size(), 4);
  auto strides = op_info->GetAttr<std::vector<int>>("strides");
  auto paddings = op_info->GetAttr<std::vector<int>>("paddings");
  auto groups = op_info->GetAttr<int>("groups");
  auto dilations = op_info->GetAttr<std::vector<int>>("dilations");
  auto fuse_relu = op_info->GetAttr<bool>("fuse_relu");
  CHECK_EQ(strides.size(), 2);
  CHECK_EQ(paddings.size(), 2);
  CHECK_EQ(dilations.size(), 2);

  // check depthwise mode
  bool depthwise_mode = input_dims[1] == groups && filter_dims[0] == groups;
  CHECK(!depthwise_mode || ((groups == 1 || groups >= 5) && dilations[0] == 1 &&
                            dilations[1] == 1))
      << "Only dilation = 1 and groups >= 5 (or groups = 1) is supported in "
         "depthwise convolution mode for NPU "
         "Convolution op";

  // create conv node and set input node from inputs_map
  auto conv_node = std::make_shared<ge::op::Convolution>(UniqueName(op_type));
  CHECK(inputs_map.count(input_var_name));
  conv_node->set_input_x(*inputs_map.at(input_var_name));
  OpList::Global().add(inputs_map.at(input_var_name));

  // add filter node
  CHECK(!inputs_map.count(filter_var_name));
  auto filter_const_node = std::make_shared<ge::op::Const>(filter_var_name);
  filter_const_node->set_attr_value(CvtFromLiteTensor(filter));
  conv_node->set_input_w(*filter_const_node);
  OpList::Global().add(filter_const_node);

  // add bias node if has bias
  if (op_info->HasInput("Bias")) {
    auto bias_var_name = op_info->Input("Bias").front();
    CHECK(!inputs_map.count(bias_var_name));
    auto* bias = scope->FindVar(bias_var_name)->GetMutable<lite::Tensor>();
    auto bias_channel_size = bias->numel();
    CHECK_EQ(bias_channel_size, filter_dims[0]);
    auto bias_const_node = std::make_shared<ge::op::Const>(bias_var_name);
    bias_const_node->set_attr_value(
        CvtFromLiteTensor(bias, {1, bias_channel_size, 1, 1}));
    conv_node->set_input_b(*bias_const_node);
    OpList::Global().add(bias_const_node);
  }

  // set attributes includes stride, kernel size, padding size, and dilation
  // size
  conv_node->set_attr_pad_mode(0);  // NOTSET
  conv_node->set_attr_group(groups);
  conv_node->set_attr_pad(ge::AttrValue::LIST_INT(
      {paddings[0], paddings[0], paddings[1], paddings[1]}));
  conv_node->set_attr_dilation(
      ge::AttrValue::LIST_INT({dilations[0], dilations[1]}));
  conv_node->set_attr_stride(ge::AttrValue::LIST_INT({strides[0], strides[1]}));
  conv_node->set_attr_kernel(
      ge::AttrValue::LIST_INT({filter_dims[2], filter_dims[3]}));

  conv_node->set_attr_mode(1);

  node_map_type outputs_map;
  if (fuse_relu) {
    // append relu node if fuse_relu is true
    auto relu_node =
        std::make_shared<ge::op::Activation>(UniqueName(op_type + "/relu"));
    relu_node->set_input_x(*conv_node);
    relu_node->set_attr_mode(1);
    OpList::Global().add(relu_node);
    outputs_map[op_info->Output("Output").front()] = relu_node;
  } else {
    outputs_map[op_info->Output("Output").front()] = conv_node;
  }
  OpList::Global().add(conv_node);
  return outputs_map;
}

node_map_type DepthwiseConvConverter(
    const std::shared_ptr<lite::OpLite> conv_op,
    const node_map_type& inputs_map) {
  VLOG(3) << "invoking DepthwiseConvConverter...";
  auto scope = conv_op->scope();
  auto op_info = conv_op->op_info();

  auto conv_node = std::make_shared<ge::op::ConvolutionDepthwise>(
      UniqueName("depthwise_conv2d"));
  auto input_var_name = op_info->Input("Input").front();
  CHECK(inputs_map.count(input_var_name));
  conv_node->set_input_x(*inputs_map.at(input_var_name));
  OpList::Global().add(inputs_map.at(input_var_name));

  // add filter node
  auto filter_var_name = op_info->Input("Filter").front();
  CHECK(!inputs_map.count(filter_var_name));
  auto filter = scope->FindVar(filter_var_name)->GetMutable<lite::Tensor>();
  auto filter_dims = filter->dims();
  CHECK_EQ(filter_dims.size(), 4);
  auto filter_const_node = std::make_shared<ge::op::Const>(filter_var_name);
  filter_const_node->set_attr_value(CvtFromLiteTensor(filter));
  conv_node->set_input_filter(*filter_const_node);
  OpList::Global().add(filter_const_node);

  // add bias node if has bias
  if (op_info->HasInput("Bias")) {
    auto bias_var_name = op_info->Input("Bias").front();
    CHECK(!inputs_map.count(bias_var_name));
    auto* bias = scope->FindVar(bias_var_name)->GetMutable<lite::Tensor>();
    auto bias_channel_size = bias->numel();
    CHECK_EQ(bias_channel_size, filter_dims[0]);
    auto bias_const_node = std::make_shared<ge::op::Const>(bias_var_name);
    bias_const_node->set_attr_value(
        CvtFromLiteTensor(bias, {1, bias_channel_size, 1, 1}));
    // conv_node->set_input_b(*bias_const_node);
    // OpList::Global().add(bias_const_node);
  }

  // set attributes includes stride, kernel size, padding size, and dilation
  // size
  auto strides = op_info->GetAttr<std::vector<int>>("strides");
  auto paddings = op_info->GetAttr<std::vector<int>>("paddings");
  auto groups = op_info->GetAttr<int>("groups");
  auto dilations = op_info->GetAttr<std::vector<int>>("dilations");
  conv_node->set_attr_pad_mode(5);  // NOTSET
  conv_node->set_attr_group(groups);
  conv_node->set_attr_pad(ge::AttrValue::LIST_INT(
      {paddings[0], paddings[0], paddings[1], paddings[1]}));
  conv_node->set_attr_dilation(
      ge::AttrValue::LIST_INT({dilations[0], dilations[1]}));
  conv_node->set_attr_stride(ge::AttrValue::LIST_INT({strides[0], strides[1]}));
  conv_node->set_attr_kernel(
      ge::AttrValue::LIST_INT({filter_dims[2], filter_dims[3]}));

  conv_node->set_attr_mode(1);
  conv_node->set_attr_algo(0);
  conv_node->set_attr_format(0);  // NCHW

  node_map_type outputs_map;
  if (op_info->GetAttr<bool>("fuse_relu")) {
    // append relu node if fuse_relu is true
    auto relu_node =
        std::make_shared<ge::op::Activation>(UniqueName("conv2d/relu"));
    relu_node->set_input_x(*conv_node);
    relu_node->set_attr_mode(1);
    OpList::Global().add(relu_node);
    outputs_map[op_info->Output("Output").front()] = relu_node;
  } else {
    outputs_map[op_info->Output("Output").front()] = conv_node;
  }
  OpList::Global().add(conv_node);
  return outputs_map;
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(conv2d, paddle::lite::npu::bridge::ConvConverter);
REGISTER_NPU_BRIDGE(depthwise_conv2d, paddle::lite::npu::bridge::ConvConverter);
// REGISTER_NPU_BRIDGE(depthwise_conv2d,
// paddle::lite::npu::bridge::DepthwiseConvConverter);
