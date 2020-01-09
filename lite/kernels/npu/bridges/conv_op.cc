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
#include "lite/kernels/npu/bridges/graph.h"
#include "lite/kernels/npu/bridges/registry.h"
#include "lite/kernels/npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace npu {

int ConvConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NPU] Converting " << op_type << "... ";

  // Get input and output vars and op attributes
  auto input_name = op_info->Input("Input").front();
  auto input_type = kernel->GetInputDeclType("Input");
  CHECK(input_type->precision() == PRECISION(kFloat));
  CHECK(input_type->layout() == DATALAYOUT(kNCHW));
  auto input = scope->FindMutableTensor(input_name);
  auto input_dims = input->dims();
  auto filter_name = op_info->Input("Filter").front();
  auto filter_type = kernel->GetInputDeclType("Filter");
  CHECK(filter_type->precision() == PRECISION(kFloat));
  CHECK(filter_type->layout() == DATALAYOUT(kNCHW));
  auto filter = scope->FindMutableTensor(filter_name);
  auto filter_dims = filter->dims();
  auto output_name = op_info->Output("Output").front();
  auto output_type = kernel->GetOutputDeclType("Output");
  CHECK(output_type->precision() == PRECISION(kFloat));
  CHECK(output_type->layout() == DATALAYOUT(kNCHW));
  auto output = scope->FindMutableTensor(output_name);
  auto output_dims = output->dims();
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

  // Input node
  std::shared_ptr<Node> input_node = nullptr;
  if (graph->Has(input_name)) {
    input_node = graph->Get(input_name);
  } else {
    input_node = graph->Add(input_name, *input);
  }

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

  // Filter node
  auto filter_node = graph->Add(filter_name, *filter);

  // Add bias node if exists bias
  // Supports the bias nodes with the following dimensions
  // 0: {oc}
  // 1: {1, oc, oh, ow}
  // 2: {n, oc, oh, ow}
  std::shared_ptr<Node> bias_node = nullptr;
  bool is_channel_bias = false;
  if (HasInputArg(op_info, scope, "Bias")) {
    auto bias_name = op_info->Input("Bias").front();
    if (graph->Has(bias_name)) {
      bias_node = graph->Get(bias_name);
    } else {
      auto bias_type = kernel->GetInputDeclType("Bias");
      CHECK(bias_type->precision() == PRECISION(kFloat));
      CHECK(bias_type->layout() == DATALAYOUT(kNCHW));
      auto bias = scope->FindMutableTensor(bias_name);
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
        LOG(WARNING)
            << "[NPU] Bias dimension " << bias_dims
            << " isn't supported in conv2d Op when output dimension is "
            << output_dims;
        return FAILED;
      }
      bias_node = graph->Add(bias_name, *bias, bias_shape);
    }
  }

  // Conv node
  std::shared_ptr<Node> conv_node = nullptr;
  if (use_depthwise_conv && is_depthwise_mode) {
    conv_node = graph->Add<ge::op::ConvolutionDepthwise>(output_name);
    auto conv_op = conv_node->data<ge::op::ConvolutionDepthwise>();
    conv_op->set_input_x(*input_node->data());
    conv_op->set_input_filter(*filter_node->data());
    conv_op->set_attr_mode(1);
    conv_op->set_attr_algo(0);
    conv_op->set_attr_format(0);    // NCHW
    conv_op->set_attr_pad_mode(5);  // VALID
    conv_op->set_attr_group(groups);
    conv_op->set_attr_pad(ge::AttrValue::LIST_INT(
        {paddings[0], paddings[1], paddings[2], paddings[3]}));
    conv_op->set_attr_dilation(
        ge::AttrValue::LIST_INT({dilations[0], dilations[1]}));
    conv_op->set_attr_stride(ge::AttrValue::LIST_INT({strides[0], strides[1]}));
    conv_op->set_attr_kernel(
        ge::AttrValue::LIST_INT({filter_dims[2], filter_dims[3]}));
    // ConvolutionDepthwise Op doesn't support bias, so append Add node to
    // support bias
    if (bias_node != nullptr) {
      auto add_node = graph->Add<ge::op::Add>(output_name);
      auto add_op = add_node->data<ge::op::Add>();
      add_op->set_input_x1(*conv_node->data());
      add_op->set_input_x2(*bias_node->data());
      conv_node = add_node;
    }
  } else {
    conv_node = graph->Add<ge::op::Convolution>(output_name);
    auto conv_op = conv_node->data<ge::op::Convolution>();
    conv_op->set_input_x(*input_node->data());
    conv_op->set_input_w(*filter_node->data());
    conv_op->set_attr_mode(1);
    conv_op->set_attr_pad_mode(0);  // NOTSET
    conv_op->set_attr_group(groups);
    conv_op->set_attr_pad(ge::AttrValue::LIST_INT(
        {paddings[0], paddings[0], paddings[2], paddings[2]}));
    conv_op->set_attr_dilation(
        ge::AttrValue::LIST_INT({dilations[0], dilations[1]}));
    conv_op->set_attr_stride(ge::AttrValue::LIST_INT({strides[0], strides[1]}));
    conv_op->set_attr_kernel(
        ge::AttrValue::LIST_INT({filter_dims[2], filter_dims[3]}));
    // Convolution Op only support bias with dimension {1, oc, 1, 1},
    // so append Add node if dimension is {1, oc, oh, ow} or (n, oc, oh, ow)
    if (bias_node != nullptr) {
      if (is_channel_bias) {
        conv_op->set_input_b(*bias_node->data());
      } else {
        auto add_node = graph->Add<ge::op::Add>(output_name);
        auto add_op = add_node->data<ge::op::Add>();
        add_op->set_input_x1(*conv_node->data());
        add_op->set_input_x2(*bias_node->data());
        conv_node = add_node;
      }
    }
  }
  CHECK(conv_node);

  if (fuse_relu) {
    // Append relu node if fuse_relu is true
    auto relu_node = graph->Add<ge::op::Activation>(output_name);
    auto relu_op = relu_node->data<ge::op::Activation>();
    relu_op->set_input_x(*conv_node->data());
    relu_op->set_attr_mode(CvtActMode("relu"));
  }
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(conv2d,
                         kNPU,
                         paddle::lite::subgraph::npu::ConvConverter);
REGISTER_SUBGRAPH_BRIDGE(depthwise_conv2d,
                         kNPU,
                         paddle::lite::subgraph::npu::ConvConverter);
