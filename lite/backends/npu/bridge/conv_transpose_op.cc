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

#include "lite/operators/conv_transpose_op.h"
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

node_map_type ConvTransposeConverter(
    const std::shared_ptr<lite::OpLite> conv_transpose_op,
    const node_map_type& inputs_map) {
  auto scope = conv_transpose_op->scope();
  auto op_info = conv_transpose_op->op_info();
  auto op_type = op_info->Type();
  auto unique_op_type = UniqueName(op_type);
  LOG(INFO) << "Converting " << op_type << "... ";

  // get input, output and op attributes
  auto input_var_name = op_info->Input("Input").front();
  auto input = scope->FindVar(input_var_name)->GetMutable<lite::Tensor>();
  auto input_shape = input->dims().Vectorize();
  auto filter_var_name = op_info->Input("Filter").front();
  auto filter = scope->FindVar(filter_var_name)->GetMutable<lite::Tensor>();
  auto filter_shape = filter->dims().Vectorize();
  CHECK_EQ(input_shape.size(), 4);
  CHECK_EQ(filter_shape.size(), 4);
  auto strides = op_info->GetAttr<std::vector<int>>("strides");
  auto paddings = op_info->GetAttr<std::vector<int>>("paddings");
  auto groups = op_info->GetAttr<int>("groups");
  auto dilations = op_info->GetAttr<std::vector<int>>("dilations");
  auto fuse_relu = op_info->GetAttr<bool>("fuse_relu");
  CHECK_EQ(strides.size(), 2);
  CHECK_EQ(paddings.size(), 2);
  CHECK_EQ(dilations.size(), 2);

  // create deconv node
  auto conv_transpose_node =
      std::make_shared<ge::op::Deconvolution>(unique_op_type);

  // create input sizes node to describe the dimensions of input tensor
  std::vector<int32_t> output_shape;
  output_shape.push_back(input_shape[0]);
  output_shape.push_back(filter_shape[1] * groups);
  for (int i = 0; i < strides.size(); i++) {
    int kernel_ext = dilations[i] * (filter_shape[i + 2] - 1) + 1;
    int output_size =
        (input_shape[i + 2] - 1) * strides[i] + kernel_ext - 2 * paddings[i];
    output_shape.push_back(output_size);
  }
  auto input_sizes_const_node =
      std::make_shared<ge::op::Const>(unique_op_type + "/input_size");
  input_sizes_const_node->set_attr_value(CreateTensorAndFillData(output_shape));
  conv_transpose_node->set_input_input_sizes(*input_sizes_const_node);
  OpList::Global().add(input_sizes_const_node);

  // create filter node
  CHECK(!inputs_map.count(filter_var_name));
  auto filter_const_node = std::make_shared<ge::op::Const>(filter_var_name);
  filter_const_node->set_attr_value(CvtFromLiteTensor(filter));
  conv_transpose_node->set_input_filter(*filter_const_node);
  OpList::Global().add(filter_const_node);

  // set input node
  CHECK(inputs_map.count(input_var_name));
  conv_transpose_node->set_input_x(*inputs_map.at(input_var_name));
  OpList::Global().add(inputs_map.at(input_var_name));

  // set attributes
  conv_transpose_node->set_attr_mode(1);
  conv_transpose_node->set_attr_format(0);    // NCHW
  conv_transpose_node->set_attr_pad_mode(0);  // NOTSET
  conv_transpose_node->set_attr_group(groups);
  conv_transpose_node->set_attr_pad(ge::AttrValue::LIST_INT(
      {paddings[0], paddings[0], paddings[1], paddings[1]}));
  conv_transpose_node->set_attr_dilation(
      ge::AttrValue::LIST_INT({dilations[0], dilations[1]}));
  conv_transpose_node->set_attr_stride(
      ge::AttrValue::LIST_INT({strides[0], strides[1]}));
  conv_transpose_node->set_attr_kernel(
      ge::AttrValue::LIST_INT({filter_shape[2], filter_shape[3]}));
  OpList::Global().add(conv_transpose_node);

  // append add node to add bias if has bias
  std::shared_ptr<ge::Operator> output_node = conv_transpose_node;
  if (HasInputArg(op_info, scope, "Bias")) {
    // create bias node
    auto bias_var_name = op_info->Input("Bias").front();
    CHECK(!inputs_map.count(bias_var_name));
    auto* bias = scope->FindVar(bias_var_name)->GetMutable<lite::Tensor>();
    auto channel_size = bias->dims().production();
    CHECK_EQ(channel_size, filter_shape[1] * groups);
    auto bias_const_node = std::make_shared<ge::op::Const>(bias_var_name);
    bias_const_node->set_attr_value(
        CvtFromLiteTensor(bias, {1, channel_size, 1, 1}));
    OpList::Global().add(bias_const_node);
    // append add node to add bias node
    auto add_node = std::make_shared<ge::op::Add>(unique_op_type + "/add");
    add_node->set_input_x1(*conv_transpose_node);
    add_node->set_input_x2(*bias_const_node);
    OpList::Global().add(add_node);
    output_node = add_node;
  }

  node_map_type outputs_map;
  if (fuse_relu) {
    // append relu node if fuse_relu is true
    auto relu_node =
        std::make_shared<ge::op::Activation>(unique_op_type + "/relu");
    relu_node->set_input_x(*output_node);
    relu_node->set_attr_mode(1);
    OpList::Global().add(relu_node);
    outputs_map[op_info->Output("Output").front()] = relu_node;
  } else {
    outputs_map[op_info->Output("Output").front()] = output_node;
  }
  return outputs_map;
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(conv2d_transpose,
                    paddle::lite::npu::bridge::ConvTransposeConverter);
