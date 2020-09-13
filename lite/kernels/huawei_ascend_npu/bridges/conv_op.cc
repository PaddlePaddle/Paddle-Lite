// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/huawei_ascend_npu/bridges/graph.h"
#include "lite/kernels/huawei_ascend_npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace huawei_ascend_npu {

int ConvConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Converting " << op_type << "... ";

  // Get input and output vars and op attributes
  auto input_name = op_info->Input("Input").front();
  auto input = scope->FindMutableTensor(input_name);
  auto input_dims = input->dims();

  auto filter_name = op_info->Input("Filter").front();
  auto filter = scope->FindMutableTensor(filter_name);
  auto filter_dims = filter->dims();

  auto output_name = op_info->Output("Output").front();
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
  bool with_act =
      op_info->HasAttr("with_act") && op_info->GetAttr<bool>("with_act");
  std::string act_type =
      with_act ? op_info->GetAttr<std::string>("act_type") : "";
  float leaky_relu_alpha = act_type == "leaky_relu"
                               ? op_info->GetAttr<float>("leaky_relu_alpha")
                               : 0.f;
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
      << "[HUAWEI_ASCEND_NPU] Paddings size should be "
         "the same or twice as the input size.";

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

  // Check Restrictions: HxW(input) == HxW(filter) if output feature h*w = 1*1
  if (output_dims[2] == 1) {
    int input_h = input_dims[2] + paddings[0] + paddings[1];
    int filter_h = (filter_dims[2] - 1) * dilations[0] + 1;
    if (input_h != filter_h) {
      LOG(WARNING) << "[HUAWEI_ASCEND_NPU] Huawei Ascend NPU DDK restriction: "
                      "input height after padding should equal to filter "
                      "height after dilation if output height is 1. Input "
                      "height after padding is: "
                   << input_h
                   << ", filter height after dilation is: " << filter_h;
      return FAILED;
    }
  }
  // Check Restrictions: HxW(input) == HxW(filter) if output feature h*w = 1*1
  if (output_dims[3] == 1) {
    int input_w = input_dims[3] + paddings[2] + paddings[3];
    int filter_w = (filter_dims[3] - 1) * dilations[1] + 1;
    if (input_w != filter_w) {
      LOG(WARNING) << "[HUAWEI_ASCEND_NPU] Huawei Ascend NPU DDK restriction: "
                      "input width after padding should equal to filter width "
                      "after dilation if output width is 1. Input width after "
                      "padding is: "
                   << input_w
                   << ", filter width after dilation is: " << filter_w;
      return FAILED;
    }
  }
  // Check Restrictions: outChannel divide groups should equal to 0
  if (oc % groups != 0) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] Huawei Ascend NPU DDK restriction: "
                    "out channel divice groups should equal to 0. out channel "
                    "is: "
                 << oc << ", groups is: " << groups;
    return FAILED;
  }

  // Filter node
  std::shared_ptr<Node> filter_node = nullptr;

  // Check depthwise mode, and decide whether use DepthwiseConv2D Op
  bool use_depthwise_conv = false;
  bool is_depthwise_mode = (ic == groups && oc == groups);
  if (is_depthwise_mode && dilations[0] == 1 && dilations[1] == 1) {
    use_depthwise_conv = true;
    // Change filter shape {oc, ic/groups = 1, kh, kw} => { K=1, oc, kh, hw}
    filter_node = graph->Add(
        filter_name, *filter, {1L, oc, filter_dims[2], filter_dims[3]});
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] DepthwiseConv2D op is used.";
  } else {
    filter_node = graph->Add(filter_name, *filter);
  }

  // Add bias node if exists bias
  // Supports the bias nodes with the following dimensions
  // 0: {oc} => 1D tensor of foramt ND
  // 1: {1, oc, oh, ow}
  // 2: {n, oc, oh, ow}
  std::vector<int64_t> bias_shape;
  std::shared_ptr<Node> bias_node = nullptr;
  bool is_channel_bias = false;
  if (HasInputArg(op_info, scope, "Bias")) {
    auto bias_name = op_info->Input("Bias").front();
    if (graph->Has(bias_name)) {
      bias_node = graph->Get(bias_name);
    } else {
      auto bias = scope->FindMutableTensor(bias_name);
      auto bias_dims = bias->dims();
      auto bias_data_size = bias_dims.production();
      auto output_data_size = output_dims.production();
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
        LOG(WARNING)
            << "[HUAWEI_ASCEND_NPU] Bias dimension " << bias_dims
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
    conv_node = graph->Add<ge::op::DepthwiseConv2D>(output_name);
    auto conv_op = conv_node->data<ge::op::DepthwiseConv2D>();
    conv_op->set_input_x(*input_node->data());
    conv_op->set_input_filter(*filter_node->data());
    conv_op->set_attr_strides(
        ge::Operator::OpListInt({1, 1, strides[0], strides[1]}));
    conv_op->set_attr_dilations({1, 1, dilations[0], dilations[1]});
    conv_op->set_attr_pads(
        {paddings[0], paddings[1], paddings[2], paddings[3]});
    conv_op->set_attr_data_format("NCHW");
    if (bias_node != nullptr && is_channel_bias) {
      conv_op->set_input_bias(*bias_node->data());
      INPUT_UPDATE(conv_op, bias, bias_node);
    }
    INPUT_UPDATE(conv_op, x, input_node);
    INPUT_UPDATE(conv_op, filter, filter_node);
    OUTPUT_UPDATE(conv_op, y, conv_node);
  } else {
    conv_node = graph->Add<ge::op::Conv2D>(output_name);
    auto conv_op = conv_node->data<ge::op::Conv2D>();
    conv_op->set_input_x(*input_node->data());
    conv_op->set_input_filter(*filter_node->data());
    conv_op->set_attr_strides(
        ge::Operator::OpListInt({1, 1, strides[0], strides[1]}));
    conv_op->set_attr_pads(ge::Operator::OpListInt(
        {paddings[0], paddings[1], paddings[2], paddings[3]}));
    conv_op->set_attr_dilations(
        ge::Operator::OpListInt({1, 1, dilations[0], dilations[1]}));
    conv_op->set_attr_groups(groups);
    conv_op->set_attr_data_format("NCHW");
    if (bias_node != nullptr && is_channel_bias) {
      conv_op->set_input_bias(*bias_node->data());
      INPUT_UPDATE(conv_op, bias, bias_node);
    }
    INPUT_UPDATE(conv_op, x, input_node);
    INPUT_UPDATE(conv_op, filter, filter_node);
    OUTPUT_UPDATE(conv_op, y, conv_node);
  }
  // append Add node to support bias
  if (bias_node != nullptr && !is_channel_bias) {
    auto add_node = graph->Add<ge::op::Add>(output_name);
    auto add_op = add_node->data<ge::op::Add>();
    add_op->set_input_x1(*conv_node->data());
    add_op->set_input_x2(*bias_node->data());
    INPUT_UPDATE(add_op, x1, conv_node);
    INPUT_UPDATE(add_op, x2, bias_node);
    OUTPUT_UPDATE(add_op, y, add_node);
  }
  CHECK(conv_node);

  // ONLY support relu/leaky_relu now
  // to do (@qili93): add more act types
  if (!act_type.empty()) {
    if (act_type == "relu") {
      auto act_node = graph->Add<ge::op::Relu>(output_name);
      auto act_op = act_node->data<ge::op::Relu>();
      act_op->set_input_x(*conv_node->data());
      INPUT_UPDATE(act_op, x, conv_node);
      OUTPUT_UPDATE(act_op, y, act_node);
    } else if (act_type == "leaky_relu") {
      auto act_node = graph->Add<ge::op::LeakyRelu>(output_name);
      auto act_op = act_node->data<ge::op::LeakyRelu>();
      act_op->set_input_x(*conv_node->data());
      act_op->set_attr_negative_slope(leaky_relu_alpha);
      INPUT_UPDATE(act_op, x, conv_node);
      OUTPUT_UPDATE(act_op, y, act_node);
    } else {
      LOG(WARNING) << "[HUAWEI_ASCEND_NPU] act type not supported: "
                   << act_type;
      return FAILED;
    }
  }

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace huawei_ascend_npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    conv2d,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::ConvConverter);
REGISTER_SUBGRAPH_BRIDGE(
    depthwise_conv2d,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::ConvConverter);
