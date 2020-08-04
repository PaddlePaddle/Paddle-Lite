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
#include <algorithm>
#include "lite/kernels/npu/bridges/registry.h"
#include "lite/kernels/rknpu/bridges/graph.h"
#include "lite/kernels/rknpu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace rknpu {

int ConvConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[RKNPU] Converting " << op_type << "... ";

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
  auto fuse_relu = op_info->GetAttr<bool>("fuse_relu");
  CHECK_EQ(strides.size(), 2L);
  CHECK_EQ(dilations.size(), 2L);
  // Check depthwise mode
  bool is_depthwise_mode = (ic == groups && oc == groups && groups != 1);
  CHECK(op_info->HasInputScale(filter_name));
  auto weight_scale = op_info->GetInputScale(filter_name);

  // for quantization
  bool enable_int8 = false;
  float input_scale = 1.0;
  float output_scale = 1.0;
  int bit_length = 8;
  DataLayoutType layout = DATALAYOUT(kNCHW);
  PrecisionType precision = PRECISION(kFloat);

  if (op_info->HasAttr("enable_int8")) {
    enable_int8 = op_info->GetAttr<bool>("enable_int8");
    CHECK(op_info->HasInputScale(input_name));
    input_scale = op_info->GetInputScale(input_name)[0];
    bit_length = op_info->GetAttr<int>("bit_length");
    CHECK(op_info->HasOutputScale(output_name));
    output_scale = op_info->GetOutputScale(output_name)[0];

    if (enable_int8) {
      precision = PRECISION(kInt8);
    }
  }

  // // Input node
  std::shared_ptr<Node> input_node = nullptr;
  if (graph->Has(input_name)) {
    input_node = graph->Get(input_name);
  } else {
    QuantizationInfo qnt;
    qnt.enable_int8 = enable_int8;

    if (enable_int8) {
      qnt.scale.clear();
      qnt.scale.push_back(input_scale);
      qnt.quant_bits = bit_length;
    }
    input_node =
        graph->Add(input_name, *input, input->precision(), layout, qnt);
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
  // Filter node
  std::shared_ptr<Node> filter_node = nullptr;
  QuantizationInfo filter_qnt;

  filter_qnt.enable_int8 = enable_int8;

  if (enable_int8) {
    filter_qnt.scale = weight_scale;
    filter_qnt.quant_bits = bit_length;
  }

  filter_node =
      graph->Add(filter_name, *filter, filter->precision(), layout, filter_qnt);

  // Add bias node if exists bias
  // Supports the bias nodes with the following dimensions
  // 0: {oc}
  std::shared_ptr<Node> bias_node = nullptr;
  if (HasInputArg(op_info, scope, "Bias")) {
    auto bias_name = op_info->Input("Bias").front();
    if (graph->Has(bias_name)) {
      bias_node = graph->Get(bias_name);
    } else {
      auto bias = scope->FindMutableTensor(bias_name);
      auto bias_dims = bias->dims();
      auto bias_data_size = bias_dims.production();
      auto output_data_size = output_dims.production();
      std::vector<int64_t> bias_shape;

      if (bias_data_size == oc) {
        // 0: {oc}
        bias_shape = {oc};
      } else {
        LOG(WARNING)
            << "[RKNPU] Bias dimension " << bias_dims
            << " isn't supported in conv2d Op when output dimension is "
            << output_dims;
        return FAILED;
      }

      if (enable_int8) {
        auto bias_name_qnt = bias_name + "/qnt";
        auto* bias_qnt = scope->NewTensor(bias_name_qnt);

        bias_qnt->Resize(bias_shape);
        bias_qnt->set_persistable(true);
        bias_qnt->set_precision(PrecisionType::kInt32);

        auto* bias_qnt_data = bias_qnt->mutable_data<int32_t>();
        auto* bias_data = bias->mutable_data<float>();

        QuantizationInfo qnt;
        qnt.enable_int8 = enable_int8;
        qnt.quant_bits = 32;

        qnt.scale.resize(weight_scale.size());
        for (int i = 0; i < weight_scale.size(); i++) {
          qnt.scale[i] = input_scale * weight_scale[i];
        }

        auto dtype_max = static_cast<int>((1 << (qnt.quant_bits - 1)) - 1);
        auto dtype_min = static_cast<int>(0 - dtype_max);

        for (int i = 0; i < oc; i++) {
          bias_qnt_data[i] =
              std::min(std::max(static_cast<int>(bias_data[i] / qnt.scale[i]),
                                dtype_min),
                       dtype_max);
        }

        bias_node = graph->Add(
            bias_name, *bias_qnt, bias_qnt->precision(), layout, qnt);
      } else {
        bias_node = graph->Add(bias_name, *bias, bias_shape);
      }
    }
  } else {
    auto bias_name = filter_name + "/bias/dummy";
    auto* bias = scope->NewTensor(bias_name);
    std::vector<int64_t> bias_shape = {oc};

    bias->Resize(bias_shape);
    bias->set_persistable(true);

    if (enable_int8) {
      bias->set_precision(PrecisionType::kInt32);
      auto* bias_data = bias->mutable_data<int32_t>();

      for (int i = 0; i < oc; i++) {
        bias_data[i] = 0;
      }

      QuantizationInfo qnt;
      qnt.enable_int8 = enable_int8;
      qnt.quant_bits = 32;
      qnt.scale.resize(weight_scale.size());
      for (int i = 0; i < weight_scale.size(); i++) {
        qnt.scale[i] = input_scale * weight_scale[i];
      }

      bias_node = graph->Add(bias_name, *bias, bias->precision(), layout, qnt);
    } else {
      bias->set_precision(PrecisionType::kFloat);
      auto* bias_data = bias->mutable_data<float>();

      for (int i = 0; i < oc; i++) {
        bias_data[i] = 0.0;
      }
      bias_node = graph->Add(bias_name, *bias, bias_shape);
    }
  }

  // Conv node
  std::shared_ptr<Node> conv_node = nullptr;
  std::shared_ptr<Node> output_node = nullptr;
  std::vector<std::shared_ptr<rk::nn::Tensor>> inputs;
  std::vector<std::shared_ptr<rk::nn::Tensor>> outputs;
  QuantizationInfo output_qnt;

  output_qnt.enable_int8 = enable_int8;

  if (enable_int8) {
    output_qnt.quant_bits = bit_length;
    output_qnt.scale.push_back(output_scale);
    output->mutable_data<int8_t>();
  }

  output_node = graph->Add(output_name, *output, precision, layout, output_qnt);

  inputs.push_back(input_node->data());
  inputs.push_back(filter_node->data());
  inputs.push_back(bias_node->data());
  outputs.push_back(output_node->data());

  rk::nn::Conv2DAttr attr;
  attr.ksize[0] = filter_dims[2];
  attr.ksize[1] = filter_dims[3];
  attr.stride[0] = strides[0];
  attr.stride[1] = strides[1];
  attr.pad[0] = paddings[0];
  attr.pad[1] = paddings[1];
  attr.pad[2] = paddings[2];
  attr.pad[3] = paddings[3];
  attr.group = groups;
  attr.weights = oc;
  attr.dilation[0] = dilations[0];
  attr.dilation[1] = dilations[1];
  attr.pad_type = rk::nn::PadType::AUTO;
  attr.has_relu = fuse_relu;

  if (is_depthwise_mode) {
    attr.multiplier = 1;
  } else {
    attr.multiplier = 0;
  }

  auto rGraph = graph->GetHandle();
  auto conv = rGraph->AddOperator(
      rk::nn::OperatorType::CONV2D, inputs, outputs, &attr, output_name);

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace rknpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(conv2d,
                         kRKNPU,
                         paddle::lite::subgraph::rknpu::ConvConverter);
REGISTER_SUBGRAPH_BRIDGE(depthwise_conv2d,
                         kRKNPU,
                         paddle::lite::subgraph::rknpu::ConvConverter);
