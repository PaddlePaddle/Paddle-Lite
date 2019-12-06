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
#include "lite/backends/npu/bridges/registry.h"
#include "lite/backends/npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridges {

int ConvConverter(cvt_ctx_type* ctx, OpLite* op) {
  auto scope = op->scope();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  VLOG(3) << "[NPU] Converting " << op_type << "... ";

  // Get input, filter and op attributes
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
  CHECK_EQ(input_dims.size(), 4L);
  CHECK_EQ(output_dims.size(), 4L);
  CHECK_EQ(filter_dims.size(), 4L);
  CHECK_EQ(output_dims[0], bs);
  CHECK_EQ(output_dims[1], oc);
  auto strides = op_info->GetAttr<std::vector<int>>("strides");
  auto paddings = op_info->GetAttr<std::vector<int>>("paddings");
  auto groups = op_info->GetAttr<int>("groups");
  auto dilations = op_info->GetAttr<std::vector<int>>("dilations");
  auto fuse_relu = op_info->GetAttr<bool>("fuse_relu");
  CHECK_EQ(strides.size(), 2L);
  CHECK_EQ(dilations.size(), 2L);

  if (paddings.size() == 2L) {
    for (size_t i = 0; i < strides.size(); ++i) {
      int copy_pad = *(paddings.begin() + 2 * i);
      paddings.insert(paddings.begin() + 2 * i + 1, copy_pad);
    }
  }
  CHECK_EQ(paddings.size(), 4L)
      << "[NPU] Paddings size should be the same or twice as the input size.";

  std::string padding_algorithm("");
  if (op_info->HasAttr("padding_algorithm")) {
    padding_algorithm = op_info->GetAttr<std::string>("padding_algorithm");
  }
  operators::UpdatePaddingAndDilation(&paddings,
                                      &dilations,
                                      strides,
                                      padding_algorithm,
                                      input_dims,
                                      filter_dims);

  // Check depthwise mode, and decide whether use ConvolutionDepthwise Op
  bool use_depthwise_conv =
      false;  // Whether use ge::op::ConvolutionDepthwise ?
  bool is_depthwise_mode = ic == groups && oc == groups;
  if (is_depthwise_mode &&
      !((groups == 1 || groups >= 5) && dilations[0] == 1 &&
        dilations[1] == 1)) {
    use_depthwise_conv = true;
    LOG(WARNING) << "[NPU] For depthwise mode, dilation = 1 and groups >= 5 "
                    "(or groups = 1) is only supported in Convolution Op, so "
                    "force to use ConvolutionDepthwise Op, but may lead poor "
                    "performance.";
  }

  // Check input node
  CHECK(ctx->HasNode(input_var_name));

  // Create filter node
  CHECK(!ctx->HasNode(filter_var_name));
  auto filter_const_node = ctx->AddNode<ge::op::Const>(filter_var_name);
  filter_const_node->set_attr_value(CvtTensor(filter));

  // Create bias node if exists bias
  // Supports the bias nodes with the following dimensions
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
      LOG(ERROR) << "[NPU] Bias dimension " << bias_dims
                 << " isn't supported in conv2d Op when output dimension is "
                 << output_dims;
    }
    if (ctx->HasNode(bias_var_name)) {
      // Bias node from input map
      bias_node = ctx->GetNode(bias_var_name);
    } else {
      // Bias node with const data
      auto bias_const_node = ctx->AddNode<ge::op::Const>(bias_var_name);
      bias_const_node->set_attr_value(CvtTensor(bias, bias_shape));
      bias_node = bias_const_node;
    }
  }

  // Create conv node and set input, filter, bias nodes and attributes
  std::shared_ptr<ge::Operator> conv_node = nullptr;
  if (use_depthwise_conv && is_depthwise_mode) {
    auto depthwise_conv_node =
        ctx->AddNode<ge::op::ConvolutionDepthwise>(output_var_name);
    depthwise_conv_node->set_input_x(*ctx->GetNode(input_var_name));
    depthwise_conv_node->set_input_filter(*filter_const_node);
    depthwise_conv_node->set_attr_mode(1);
    depthwise_conv_node->set_attr_algo(0);
    depthwise_conv_node->set_attr_format(0);    // NCHW
    depthwise_conv_node->set_attr_pad_mode(5);  // VALID
    depthwise_conv_node->set_attr_group(groups);
    depthwise_conv_node->set_attr_pad(ge::AttrValue::LIST_INT(
        {paddings[0], paddings[1], paddings[2], paddings[3]}));
    depthwise_conv_node->set_attr_dilation(
        ge::AttrValue::LIST_INT({dilations[0], dilations[1]}));
    depthwise_conv_node->set_attr_stride(
        ge::AttrValue::LIST_INT({strides[0], strides[1]}));
    depthwise_conv_node->set_attr_kernel(
        ge::AttrValue::LIST_INT({filter_dims[2], filter_dims[3]}));
    conv_node = depthwise_conv_node;
    // ConvolutionDepthwise Op doesn't support bias, so append Add node to
    // support bias
    if (bias_node != nullptr) {
      auto add_node = ctx->AddNode<ge::op::Add>(output_var_name);
      add_node->set_input_x1(*depthwise_conv_node);
      add_node->set_input_x2(*bias_node);
      conv_node = add_node;
    }
  } else {
    auto common_conv_node = ctx->AddNode<ge::op::Convolution>(output_var_name);
    common_conv_node->set_input_x(*ctx->GetNode(input_var_name));
    common_conv_node->set_input_w(*filter_const_node);
    common_conv_node->set_attr_mode(1);
    common_conv_node->set_attr_pad_mode(0);  // NOTSET
    common_conv_node->set_attr_group(groups);
    common_conv_node->set_attr_pad(ge::AttrValue::LIST_INT(
        {paddings[0], paddings[0], paddings[2], paddings[2]}));
    common_conv_node->set_attr_dilation(
        ge::AttrValue::LIST_INT({dilations[0], dilations[1]}));
    common_conv_node->set_attr_stride(
        ge::AttrValue::LIST_INT({strides[0], strides[1]}));
    common_conv_node->set_attr_kernel(
        ge::AttrValue::LIST_INT({filter_dims[2], filter_dims[3]}));
    conv_node = common_conv_node;
    // Convolution Op only support bias with dimension {1, oc, 1, 1},
    // so append Add node if dimension is {1, oc, oh, ow} or (n, oc, oh, ow)
    if (bias_node != nullptr) {
      if (is_channel_bias) {
        common_conv_node->set_input_b(*bias_node);
      } else {
        auto add_node = ctx->AddNode<ge::op::Add>(output_var_name);
        add_node->set_input_x1(*common_conv_node);
        add_node->set_input_x2(*bias_node);
        conv_node = add_node;
      }
    }
  }
  CHECK(conv_node);

  if (fuse_relu) {
    // Append relu node if fuse_relu is true
    auto relu_node = ctx->AddNode<ge::op::Activation>(output_var_name);
    relu_node->set_input_x(*conv_node);
    relu_node->set_attr_mode(CvtActMode("relu"));
  }
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace bridges
}  // namespace npu
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(conv2d, paddle::lite::npu::bridges::ConvConverter);
REGISTER_NPU_BRIDGE(depthwise_conv2d,
                    paddle::lite::npu::bridges::ConvConverter);
