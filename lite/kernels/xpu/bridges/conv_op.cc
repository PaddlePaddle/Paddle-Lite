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

#include "lite/backends/xpu/builder.h"
#include "lite/kernels/xpu/bridges/registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {
namespace bridges {

node_map_type ConvConverter(const std::shared_ptr<lite::OpLite> conv_op,
                            const node_map_type& inputs_map) {
  auto scope = conv_op->scope();
  auto op_info = conv_op->op_info();
  auto op_type = op_info->Type();
  auto unique_op_type = lite::xpu::UniqueName(op_type);
  LOG(INFO) << "Converting " << op_type << "... ";

  // get input, filter and op attributes
  auto input_var_name = op_info->Input("Input").front();
  auto input = scope->FindVar(input_var_name)->GetMutable<lite::Tensor>();
  auto input_dims = input->dims();
  auto filter_var_name = op_info->Input("Filter").front();
  auto filter = scope->FindVar(filter_var_name)->GetMutable<lite::Tensor>();
  auto filter_dims = filter->dims();
  auto bs = input_dims[0];
  auto oc = filter_dims[0];
  CHECK_EQ(input_dims.size(), 4);
  CHECK_EQ(filter_dims.size(), 4);
  auto strides = op_info->GetAttr<std::vector<int>>("strides");
  auto paddings = op_info->GetAttr<std::vector<int>>("paddings");
  auto groups = op_info->GetAttr<int>("groups");
  auto dilations = op_info->GetAttr<std::vector<int>>("dilations");
  auto fuse_relu = op_info->GetAttr<bool>("fuse_relu");
  CHECK_EQ(strides.size(), 2);
  CHECK_EQ(paddings.size(), 2);
  CHECK_EQ(dilations.size(), 2);
  std::vector<int64_t> output_shape({bs, oc});
  for (size_t i = 0; i < 2; i++) {
    const int dkernel = dilations[i] * (filter_dims[2 + i] - 1) + 1;
    output_shape.push_back(
        (input_dims[i + 2] + 2 * paddings[i] - dkernel) / strides[i] + 1);
  }
  DDim output_dims(output_shape);

  // check network context
  CHECK(inputs_map.network_builder != nullptr);
  CHECK(inputs_map.const_tensors != nullptr);

  // create filter node
  CHECK(!inputs_map.output_nodes.count(filter_var_name));
  auto filter_const_node = std::make_shared<xtcl::xExpr>(
      inputs_map.network_builder->CreateTensor(filter_var_name,
                                               lite::xpu::CvtShape(filter_dims),
                                               ::xtcl::Float(32)));
  auto filter_const_tensor = lite::xpu::CvtTensor(filter);
  inputs_map.const_tensors->emplace(
      std::make_pair(filter_var_name, *filter_const_tensor));

  // create conv node and set input, filter, bias nodes and attributes
  auto conv_attrs = xtcl::make_node<xtcl::network::Conv2DAttrs>();
  conv_attrs->strides = std::move(lite::xpu::CvtShape(strides));
  conv_attrs->padding = std::move(lite::xpu::CvtShape(paddings));
  conv_attrs->dilation = std::move(lite::xpu::CvtShape(dilations));
  conv_attrs->groups = groups;
  // conv_attrs->channels = nullptr;
  conv_attrs->kernel_size = std::move(xtcl::Array<xtcl::xIndexExpr>(nullptr));
  conv_attrs->data_layout = "NCHW";
  conv_attrs->kernel_layout = "OIHW";
  conv_attrs->out_layout = "";
  // conv_attrs->out_dtype = "";
  CHECK(inputs_map.output_nodes.count(input_var_name));
  auto conv_node =
      std::make_shared<xtcl::xExpr>(inputs_map.network_builder->CreateConv2D(
          *inputs_map.output_nodes.at(input_var_name),
          *filter_const_node,
          conv_attrs));
  inputs_map.network_builder->SetLayer(unique_op_type);

  // create bias node if has bias
  // supports the bias nodes with the following dimensions
  // 0: {oc}
  // 1: {1, oc, oh, ow}
  // 2: {n, oc, oh, ow}
  if (lite::xpu::HasInputArg(op_info, scope, "Bias")) {
    auto bias_var_name = op_info->Input("Bias").front();
    auto* bias = scope->FindVar(bias_var_name)->GetMutable<lite::Tensor>();
    auto bias_dims = bias->dims();
    auto bias_data_size = bias_dims.production();
    auto output_data_size = output_dims.production();
    std::vector<int64_t> bias_shape;
    bool is_channel_bias = false;
    if (bias_data_size == oc) {
      // 0: {oc}
      bias_shape = {oc};
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
    std::shared_ptr<xtcl::xExpr> bias_node = nullptr;
    if (inputs_map.output_nodes.count(bias_var_name)) {
      // bias node from input map
      bias_node = inputs_map.output_nodes.at(bias_var_name);
    } else {
      // bias node with const data
      auto bias_const_node = std::make_shared<xtcl::xExpr>(
          inputs_map.network_builder->CreateTensor(
              bias_var_name,
              lite::xpu::CvtShape(bias_shape),
              ::xtcl::Float(32)));
      auto bias_const_tensor = lite::xpu::CvtTensor(bias, bias_shape);
      inputs_map.const_tensors->emplace(
          std::make_pair(bias_var_name, *bias_const_tensor));
      bias_node = bias_const_node;
    }
    std::shared_ptr<xtcl::xExpr> add_node = nullptr;
    if (is_channel_bias) {
      add_node = std::make_shared<xtcl::xExpr>(
          inputs_map.network_builder->CreateBiasAdd(*conv_node, *bias_node, 1));
    } else {
      add_node = std::make_shared<xtcl::xExpr>(
          inputs_map.network_builder->CreateBinaryOp(
              "add", *conv_node, *bias_node));
    }
    inputs_map.network_builder->SetLayer(unique_op_type + "/add");
    conv_node = add_node;
  }

  node_map_type outputs_map;
  outputs_map.network_builder = inputs_map.network_builder;
  outputs_map.const_tensors = inputs_map.const_tensors;
  if (fuse_relu) {
    // append relu node if fuse_relu is true
    auto relu_node = std::make_shared<xtcl::xExpr>(
        inputs_map.network_builder->CreateRelu(*conv_node));
    inputs_map.network_builder->SetLayer(unique_op_type + "/relu");
    outputs_map.output_nodes[op_info->Output("Output").front()] = relu_node;
  } else {
    outputs_map.output_nodes[op_info->Output("Output").front()] = conv_node;
  }
  return outputs_map;
}

}  // namespace bridges
}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_XPU_BRIDGE(conv2d, paddle::lite::kernels::xpu::bridges::ConvConverter);
REGISTER_XPU_BRIDGE(depthwise_conv2d,
                    paddle::lite::kernels::xpu::bridges::ConvConverter);
