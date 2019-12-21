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

#include "lite/kernels/npu/bridges/registry.h"
#include "lite/kernels/xpu/bridges/graph.h"
#include "lite/kernels/xpu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace xpu {

inline void UpdatePaddingAndDilation(std::vector<int>* paddings,
                                     std::vector<int>* dilations,
                                     const std::vector<int>& strides,
                                     const std::string padding_algorithm,
                                     const lite::DDim data_dims,
                                     const lite::DDim& ksize) {
  // when padding_desc is "VALID" or "SAME"
  if (padding_algorithm == "SAME") {
    for (size_t i = 0; i < strides.size(); ++i) {
      int out_size = (data_dims[i + 2] + strides[i] - 1) / strides[i];
      int pad_sum = std::max(
          (out_size - 1) * strides[i] + ksize[i + 2] - data_dims[i + 2],
          (int64_t)0);
      int pad_0 = pad_sum / 2;
      int pad_1 = pad_sum - pad_0;
      // pad
      *(paddings->begin() + i * 2) = pad_0;
      *(paddings->begin() + i * 2 + 1) = pad_1;
      // dilation
      *(dilations->begin() + i) = 1;
    }
  } else if (padding_algorithm == "VALID") {
    for (auto& it : *paddings) {
      it = 0;
    }
  }
}

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
  auto bs = input_dims[0];
  auto oc = filter_dims[0];
  CHECK_EQ(input_dims.size(), 4);
  CHECK_EQ(filter_dims.size(), 4);
  auto strides = op_info->GetAttr<std::vector<int>>("strides");
  auto paddings = op_info->GetAttr<std::vector<int>>("paddings");
  auto groups = op_info->GetAttr<int>("groups");
  auto dilations = op_info->GetAttr<std::vector<int>>("dilations");
  auto fuse_relu = op_info->GetAttr<bool>("fuse_relu");
  CHECK_EQ(strides.size(), 2L);
  CHECK_EQ(dilations.size(), 2L);

  // Input node
  std::shared_ptr<xtcl::xExpr> input_node = nullptr;
  if (graph->HasNode(input_name)) {
    input_node = graph->GetNode(input_name);
  } else {
    input_node = graph->AddNode(input_name, input_dims);
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
  UpdatePaddingAndDilation(&paddings,
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
  auto filter_const_node = graph->AddNode(filter_name, *filter);

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
      graph->AddNode(output_name,
                     graph->builder_.CreateConv2D(
                         *input_node, *filter_const_node, conv_attrs));

  // Add bias node if exists bias
  // supports the bias nodes with the following dimensions
  // 0: {oc}
  // 1: {1, oc, oh, ow}
  // 2: {n, oc, oh, ow}
  if (HasInputArg(op_info, scope, "Bias")) {
    auto bias_name = op_info->Input("Bias").front();
    auto bias_type = kernel->GetInputDeclType("Bias");
    CHECK(bias_type->precision() == PRECISION(kFloat));
    CHECK(bias_type->layout() == DATALAYOUT(kNCHW));
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
    std::shared_ptr<xtcl::xExpr> bias_node = nullptr;
    if (graph->HasNode(bias_name)) {
      // Bias node from input node
      bias_node = graph->GetNode(bias_name);
    } else {
      // Bias node with const data
      bias_node = graph->AddNode(bias_name, *bias, bias_shape);
    }
    std::shared_ptr<xtcl::xExpr> add_node = nullptr;
    if (is_channel_bias) {
      add_node = graph->AddNode(
          output_name,
          graph->builder_.CreateBiasAdd(*conv_node, 1, *bias_node));
    } else {
      add_node = graph->AddNode(
          output_name,
          graph->builder_.CreateBinaryOp("add", *conv_node, *bias_node));
    }
    conv_node = add_node;
  }

  if (fuse_relu) {
    // Append relu node if fuse_relu is true
    graph->AddNode(output_name, graph->builder_.CreateRelu(*conv_node));
  }
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace xpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(XPU,
                         conv2d,
                         paddle::lite::subgraph::xpu::ConvConverter);
REGISTER_SUBGRAPH_BRIDGE(XPU,
                         depthwise_conv2d,
                         paddle::lite::subgraph::xpu::ConvConverter);
