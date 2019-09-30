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
#include "lite/backends/npu/bridge/registry.h"
#include "lite/backends/npu/bridge/utils.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

node_map_type ConvConverter(const std::shared_ptr<lite::OpLite> conv_op,
                            const node_map_type& inputs_map) {
  auto scope = conv_op->scope();
  auto op_info = conv_op->op_info();
  auto op_type = op_info->Type();
  auto unique_op_type = UniqueName(op_type);
  LOG(INFO) << "Converting " << op_type << "... ";

  // get input, filter and op attributes
  auto input_var_name = op_info->Input("Input").front();
  auto input = scope->FindVar(input_var_name)->GetMutable<lite::Tensor>();
  auto input_dims = input->dims();
  auto output_var_name = op_info->Output("Output").front();
  auto output = scope->FindVar(output_var_name)->GetMutable<lite::Tensor>();
  auto output_dims = output->dims();
  auto filter_var_name = op_info->Input("Filter").front();
  auto filter = scope->FindVar(filter_var_name)->GetMutable<lite::Tensor>();
  auto filter_dims = filter->dims();
  auto bs = input_dims[0];
  auto ic = input_dims[1];
  auto oc = filter_dims[0];
  CHECK_EQ(input_dims.size(), 4);
  CHECK_EQ(output_dims.size(), 4);
  CHECK_EQ(filter_dims.size(), 4);
  CHECK_EQ(output_dims[0], bs);
  CHECK_EQ(output_dims[1], oc);
  auto strides = op_info->GetAttr<std::vector<int>>("strides");
  auto paddings = op_info->GetAttr<std::vector<int>>("paddings");
  auto groups = op_info->GetAttr<int>("groups");
  auto dilations = op_info->GetAttr<std::vector<int>>("dilations");
  auto fuse_relu = op_info->GetAttr<bool>("fuse_relu");
  CHECK_EQ(strides.size(), 2);
  CHECK_EQ(paddings.size(), 2);
  CHECK_EQ(dilations.size(), 2);

  // check depthwise mode, and decide whether use ConvolutionDepthwise Op
  bool use_depthwise_conv =
      false;  // whether use ge::op::ConvolutionDepthwise ?
  bool is_depthwise_mode = ic == groups && oc == groups;
  if (is_depthwise_mode &&
      !((groups == 1 || groups >= 5) && dilations[0] == 1 &&
        dilations[1] == 1)) {
    use_depthwise_conv = true;
    LOG(WARNING) << "For depthwise mode, dilation = 1 and groups >= 5 (or "
                    "groups = 1) is only supported in "
                    "Convolution Op, so force to use ConvolutionDepthwise Op, "
                    "but may lead poor performance.";
  }

  // check input
  CHECK(inputs_map.count(input_var_name));
  OpList::Global().add(inputs_map.at(input_var_name));

  // create filter node
  CHECK(!inputs_map.count(filter_var_name));
  auto filter_const_node = std::make_shared<ge::op::Const>(filter_var_name);
  filter_const_node->set_attr_value(CvtFromLiteTensor(filter));
  OpList::Global().add(filter_const_node);

  // create bias node if has bias
  // supports the bias nodes with the following dimensions
  // 0: {oc}
  // 1: {1, oc, oh, ow}
  // 2: {n, oc, oh, ow}
  std::shared_ptr<ge::Operator> bias_node = nullptr;
  bool is_channel_bias = false;
  if (HasInputArg(op_info, scope, "Bias")) {
    auto bias_var_name = op_info->Input("Bias").front();
    auto* bias = scope->FindVar(bias_var_name)->GetMutable<lite::Tensor>();
    auto bias_dims = bias->dims();
    auto bias_data_size = bias_dims.production();
    auto output_data_size = output_dims.production();
    std::vector<int64_t> bias_shape;
    if (bias_data_size == oc) {
      // 0: {oc}
      bias_shape = {1, oc, 1, 1};
      is_channel_bias = true;
    } else if (bias_data_size == output_data_size / bs) {
      // 1: {1, oc, oh, ow}
      bias_shape = {1, output_dims[1], output_dims[2], output_dims[3]};
    } else if (bias_data_size == output_data_size) {
      // 2: {n, oc, oh, ow}
      bias_shape = output_dims.Vectorize();
    } else {
      LOG(ERROR) << "bias dimension " << bias_dims
                 << " isn't supported in conv2d Op when output dimension is "
                 << output_dims;
    }
    if (inputs_map.count(bias_var_name)) {
      // bias node from input map
      bias_node = inputs_map.at(bias_var_name);
    } else {
      // bias node with const data
      auto bias_const_node = std::make_shared<ge::op::Const>(bias_var_name);
      bias_const_node->set_attr_value(CvtFromLiteTensor(bias, bias_shape));
      bias_node = bias_const_node;
    }
    OpList::Global().add(bias_node);
  }

  // create conv node and set input, filter, bias nodes and attributes
  std::shared_ptr<ge::Operator> conv_node = nullptr;
  if (use_depthwise_conv && is_depthwise_mode) {
    auto depthwise_conv_node =
        std::make_shared<ge::op::ConvolutionDepthwise>(unique_op_type);
    depthwise_conv_node->set_input_x(*inputs_map.at(input_var_name));
    depthwise_conv_node->set_input_filter(*filter_const_node);
    depthwise_conv_node->set_attr_mode(1);
    depthwise_conv_node->set_attr_algo(0);
    depthwise_conv_node->set_attr_format(0);    // NCHW
    depthwise_conv_node->set_attr_pad_mode(5);  // VALID
    depthwise_conv_node->set_attr_group(groups);
    depthwise_conv_node->set_attr_pad(ge::AttrValue::LIST_INT(
        {paddings[0], paddings[0], paddings[1], paddings[1]}));
    depthwise_conv_node->set_attr_dilation(
        ge::AttrValue::LIST_INT({dilations[0], dilations[1]}));
    depthwise_conv_node->set_attr_stride(
        ge::AttrValue::LIST_INT({strides[0], strides[1]}));
    depthwise_conv_node->set_attr_kernel(
        ge::AttrValue::LIST_INT({filter_dims[2], filter_dims[3]}));
    OpList::Global().add(depthwise_conv_node);
    conv_node = depthwise_conv_node;
    // ConvolutionDepthwise Op doesn't support bias, so append Add node to
    // support bias
    if (bias_node != nullptr) {
      auto add_node = std::make_shared<ge::op::Add>(unique_op_type + "/add");
      add_node->set_input_x1(*depthwise_conv_node);
      add_node->set_input_x2(*bias_node);
      OpList::Global().add(add_node);
      conv_node = add_node;
    }
  } else {
    auto common_conv_node =
        std::make_shared<ge::op::Convolution>(unique_op_type);
    common_conv_node->set_input_x(*inputs_map.at(input_var_name));
    common_conv_node->set_input_w(*filter_const_node);
    common_conv_node->set_attr_mode(1);
    common_conv_node->set_attr_pad_mode(0);  // NOTSET
    common_conv_node->set_attr_group(groups);
    common_conv_node->set_attr_pad(ge::AttrValue::LIST_INT(
        {paddings[0], paddings[0], paddings[1], paddings[1]}));
    common_conv_node->set_attr_dilation(
        ge::AttrValue::LIST_INT({dilations[0], dilations[1]}));
    common_conv_node->set_attr_stride(
        ge::AttrValue::LIST_INT({strides[0], strides[1]}));
    common_conv_node->set_attr_kernel(
        ge::AttrValue::LIST_INT({filter_dims[2], filter_dims[3]}));
    OpList::Global().add(common_conv_node);
    conv_node = common_conv_node;
    // Convolution Op only support bias with dimension {1, oc, 1, 1},
    // so append Add node if dimension is {1, oc, oh, ow} or (n, oc, oh, ow)
    if (bias_node != nullptr) {
      if (is_channel_bias) {
        common_conv_node->set_input_b(*bias_node);
      } else {
        auto add_node = std::make_shared<ge::op::Add>(unique_op_type + "/add");
        add_node->set_input_x1(*common_conv_node);
        add_node->set_input_x2(*bias_node);
        OpList::Global().add(add_node);
        conv_node = add_node;
      }
    }
  }
  CHECK(conv_node);

  node_map_type outputs_map;
  if (fuse_relu) {
    // append relu node if fuse_relu is true
    auto relu_node =
        std::make_shared<ge::op::Activation>(unique_op_type + "/relu");
    relu_node->set_input_x(*conv_node);
    relu_node->set_attr_mode(1);
    OpList::Global().add(relu_node);
    outputs_map[op_info->Output("Output").front()] = relu_node;
  } else {
    outputs_map[op_info->Output("Output").front()] = conv_node;
  }
  return outputs_map;
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(conv2d, paddle::lite::npu::bridge::ConvConverter);
REGISTER_NPU_BRIDGE(depthwise_conv2d, paddle::lite::npu::bridge::ConvConverter);
