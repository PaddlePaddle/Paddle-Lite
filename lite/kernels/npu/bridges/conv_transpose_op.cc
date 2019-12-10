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

#include "lite/core/mir/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/npu/bridges/graph.h"
#include "lite/kernels/npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace npu {

int ConvTransposeConverter(void* ctx, OpLite* op) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NPU] Converting " << op_type << "... ";

  // Get input, output and op attributes
  auto input_var_name = op_info->Input("Input").front();
  auto input = scope->FindVar(input_var_name)->GetMutable<Tensor>();
  auto input_shape = input->dims().Vectorize();
  auto output_var_name = op_info->Output("Output").front();
  auto filter_var_name = op_info->Input("Filter").front();
  auto filter = scope->FindVar(filter_var_name)->GetMutable<Tensor>();
  auto filter_shape = filter->dims().Vectorize();
  CHECK_EQ(input_shape.size(), 4);
  CHECK_EQ(filter_shape.size(), 4);
  auto strides = op_info->GetAttr<std::vector<int>>("strides");
  auto paddings = op_info->GetAttr<std::vector<int>>("paddings");
  auto groups = op_info->GetAttr<int>("groups");
  auto dilations = op_info->GetAttr<std::vector<int>>("dilations");
  auto fuse_relu = op_info->GetAttr<bool>("fuse_relu");
  CHECK_EQ(strides.size(), 2L);
  CHECK_EQ(dilations.size(), 2L);

  if (paddings.size() == 2L) {
    for (size_t i = 0; i < 2L; ++i) {
      int copy_pad = *(paddings.begin() + 2 * i);
      paddings.insert(paddings.begin() + 2 * i + 1, copy_pad);
    }
  }
  CHECK_EQ(paddings.size(), 4L)
      << "[NPU] Paddings size should be the same or twice as the input size.";

  // Create deconv node
  auto conv_transpose_node =
      graph->AddNode<ge::op::Deconvolution>(output_var_name);

  // Create input sizes node to describe the dimensions of input tensor
  std::vector<int32_t> input_sizes;
  input_sizes.push_back(input_shape[0]);
  input_sizes.push_back(filter_shape[1] * groups);
  for (int i = 0; i < strides.size(); i++) {
    int kernel_ext = dilations[i] * (filter_shape[i + 2] - 1) + 1;
    int output_size =
        (input_shape[i + 2] - 1) * strides[i] + kernel_ext - 2 * paddings[i];
    input_sizes.push_back(output_size);
  }
  auto input_sizes_const_node =
      graph->AddNode(output_var_name + "/input_sizes", input_sizes);
  conv_transpose_node->set_input_input_sizes(*input_sizes_const_node);

  // Create filter node
  auto filter_const_node = graph->AddNode(filter_var_name, *filter);
  conv_transpose_node->set_input_filter(*filter_const_node);

  // Set input node
  conv_transpose_node->set_input_x(*graph->GetNode(input_var_name));

  // Set attributes
  conv_transpose_node->set_attr_format(0);    // NCHW
  conv_transpose_node->set_attr_pad_mode(0);  // NOTSET
  conv_transpose_node->set_attr_group(groups);
  conv_transpose_node->set_attr_pad(ge::AttrValue::LIST_INT(
      {paddings[0], paddings[1], paddings[2], paddings[3]}));
  conv_transpose_node->set_attr_dilation(
      ge::AttrValue::LIST_INT({dilations[0], dilations[1]}));
  conv_transpose_node->set_attr_stride(
      ge::AttrValue::LIST_INT({strides[0], strides[1]}));
  conv_transpose_node->set_attr_kernel(
      ge::AttrValue::LIST_INT({filter_shape[2], filter_shape[3]}));

  // Append add node to add bias if exists bias
  std::shared_ptr<ge::Operator> output_node = conv_transpose_node;
  if (HasInputArg(op_info, scope, "Bias")) {
    // Create bias node
    auto bias_var_name = op_info->Input("Bias").front();
    CHECK(!graph->HasNode(bias_var_name));
    auto* bias = scope->FindVar(bias_var_name)->GetMutable<Tensor>();
    auto channel_size = bias->dims().production();
    CHECK_EQ(channel_size, filter_shape[1] * groups);
    auto bias_const_node =
        graph->AddNode(bias_var_name, *bias, {1, channel_size, 1, 1});
    // Append add node to add bias node
    auto add_node = graph->AddNode<ge::op::Add>(output_var_name);
    add_node->set_input_x1(*conv_transpose_node);
    add_node->set_input_x2(*bias_const_node);
    output_node = add_node;
  }

  if (fuse_relu) {
    // Append relu node if fuse_relu is true
    auto relu_node = graph->AddNode<ge::op::Activation>(output_var_name);
    relu_node->set_input_x(*output_node);
    relu_node->set_attr_mode(CvtActMode("relu"));
  }
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(NPU,
                         conv2d_transpose,
                         paddle::lite::subgraph::npu::ConvTransposeConverter);
