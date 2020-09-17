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
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/xpu/bridges/graph.h"
#include "lite/kernels/xpu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace xpu {

int ConvConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[XPU] Converting " << op_type << "... ";

  // Get input and output vars and op attributes
  auto input_name = op_info->Input("Input").front();
  auto input = scope->FindMutableTensor(input_name);
  auto input_dims = input->dims();
  auto filter_name = op_info->Input("Filter").front();
  auto filter = scope->FindMutableTensor(filter_name);
  auto filter_dims = filter->dims();
  auto output_name = op_info->Output("Output").front();
  auto bs = input_dims[0];
  auto oc = filter_dims[0];
  CHECK_EQ(input_dims.size(), 4);
  CHECK_EQ(filter_dims.size(), 4);
  auto strides = op_info->GetAttr<std::vector<int>>("strides");
  std::vector<int> paddings = op_info->GetAttr<std::vector<int>>("paddings");
  auto groups = op_info->GetAttr<int>("groups");
  std::vector<int> dilations = op_info->GetAttr<std::vector<int>>("dilations");
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
      << "Paddings size should be the same or twice as the input size.";

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

  std::vector<int64_t> output_shape({bs, oc});
  for (size_t i = 0; i < 2; i++) {
    const int dkernel = dilations[i] * (filter_dims[2 + i] - 1) + 1;
    output_shape.push_back(
        (input_dims[i + 2] + paddings[2 * i] + paddings[2 * i + 1] - dkernel) /
            strides[i] +
        1);
  }
  DDim output_dims(output_shape);

  // Filter node
  auto filter_node = graph->Add(filter_name, *filter);

  // Conv node
  auto conv_attrs = xtcl::make_node<xtcl::network::Conv2DAttrs>();
  conv_attrs->strides = std::move(CvtShape<xtcl::xIndexExpr>(strides));
  conv_attrs->padding = std::move(CvtShape<xtcl::xIndexExpr>(paddings));
  conv_attrs->dilation = std::move(CvtShape<xtcl::xIndexExpr>(dilations));
  conv_attrs->groups = groups;
  // conv_attrs->channels = nullptr;
  conv_attrs->kernel_size = std::move(xtcl::Array<xtcl::xIndexExpr>(nullptr));
  conv_attrs->data_layout = "NCHW";
  conv_attrs->kernel_layout = "OIHW";
  conv_attrs->out_layout = "";
  // conv_attrs->out_dtype = "";
  auto conv_node =
      graph->Add(output_name,
                 graph->builder_.CreateConv2D(
                     *input_node->data(), *filter_node->data(), conv_attrs));

  // Add bias node if exists bias
  // supports the bias nodes with the following dimensions
  // 0: {oc}
  // 1: {1, oc, oh, ow}
  // 2: {n, oc, oh, ow}
  if (HasInputArg(op_info, scope, "Bias")) {
    auto bias_name = op_info->Input("Bias").front();
    auto bias = scope->FindMutableTensor(bias_name);
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
      LOG(ERROR) << "[XPU] Bias dimension " << bias_dims
                 << " isn't supported in conv2d Op when output dimension is "
                 << output_dims;
    }
    std::shared_ptr<Node> bias_node = nullptr;
    if (graph->Has(bias_name)) {
      bias_node = graph->Get(bias_name);
    } else {
      bias_node = graph->Add(bias_name, *bias, bias_shape);
    }
    if (is_channel_bias) {
      conv_node = graph->Add(output_name,
                             graph->builder_.CreateBiasAdd(
                                 *conv_node->data(), 1, *bias_node->data()));
    } else {
      conv_node =
          graph->Add(output_name,
                     graph->builder_.CreateBinaryOp(
                         "add", *conv_node->data(), *bias_node->data()));
    }
  }

  if (fuse_relu) {
    // Append relu node if fuse_relu is true
    graph->Add(output_name, graph->builder_.CreateRelu(*conv_node->data()));
  }
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace xpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(conv2d,
                         kXPU,
                         paddle::lite::subgraph::xpu::ConvConverter);
REGISTER_SUBGRAPH_BRIDGE(depthwise_conv2d,
                         kXPU,
                         paddle::lite::subgraph::xpu::ConvConverter);
