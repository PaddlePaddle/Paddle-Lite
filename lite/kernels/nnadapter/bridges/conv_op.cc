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
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/nnadapter/bridges/graph.h"
#include "lite/kernels/nnadapter/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace nnadapter {

int ConvConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto handle = graph->Handle();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NNAdapter] Converting " << op_type << "... ";

  // Get input and output vars and op attributes
  auto input_name = op_info->Input("Input").front();
  auto input_scale_name = "Input0_scale";
  auto input = scope->FindMutableTensor(input_name);
  auto input_dims = input->dims();
  auto filter_name = op_info->Input("Filter").front();
  auto filter_scale_name = "Filter0_scale";
  auto filter = scope->FindMutableTensor(filter_name);
  auto filter_dims = filter->dims();
  auto output_name = op_info->Output("Output").front();
  auto output_scale_name = "Output0_scale";
  auto output = scope->FindMutableTensor(output_name);
  auto output_dims = output->dims();
  auto batch_size = input_dims[0];
  auto input_channel_size = input_dims[1];
  auto output_channel_size = filter_dims[0];
  CHECK_EQ(input_dims.size(), 4L);
  CHECK_EQ(output_dims.size(), 4L);
  CHECK_EQ(filter_dims.size(), 4L);
  CHECK_EQ(output_dims[0], batch_size);
  CHECK_EQ(output_dims[1], output_channel_size);
  auto strides = op_info->GetAttr<std::vector<int>>("strides");
  std::vector<int> paddings = op_info->GetAttr<std::vector<int>>("paddings");
  auto groups = op_info->GetAttr<int>("groups");
  std::vector<int> dilations = op_info->GetAttr<std::vector<int>>("dilations");
  bool with_act =
      op_info->HasAttr("with_act") && op_info->GetAttr<bool>("with_act");
  std::string act_type =
      with_act ? op_info->GetAttr<std::string>("act_type") : "";
  float leaky_relu_alpha = act_type == "leaky_relu"
                               ? op_info->GetAttr<float>("leaky_relu_alpha")
                               : 0.f;
  CHECK_EQ(strides.size(), 2L);
  CHECK_EQ(dilations.size(), 2L);
  if (paddings.size() == 2L) {
    for (size_t i = 0; i < strides.size(); i++) {
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
  auto fuse_relu = op_info->GetAttr<bool>("fuse_relu");
  // Check depthwise mode
  bool is_depthwise_mode = (input_channel_size == groups &&
                            output_channel_size == groups && groups != 1);
  CHECK(!is_depthwise_mode) << "depthwise mode is not supported.";

  CHECK(op_info->HasOutputScale(output_scale_name, true));
  auto output_scale = op_info->GetOutputScale(output_scale_name, true)[0];

  // Input operand
  CHECK(op_info->HasInputScale(input_scale_name, true));
  auto input_scale = op_info->GetInputScale(input_scale_name, true)[0];
  std::shared_ptr<Node> input_node = nullptr;
  if (graph->Has(input_name)) {
    input_node = graph->Get(input_name);
  } else {
    NNAdapterOperand* input_operand;
    NNAdapterOperandType input_type;
    input_type.precision = NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER;
    input_type.symmPerLayerParams.scale = input_scale;
    input_type.dimensionCount = input_dims.size();
    input_type.dimensions[0] = static_cast<uint32_t>(input_dims[0]);
    input_type.dimensions[1] = static_cast<uint32_t>(input_dims[1]);
    input_type.dimensions[2] = static_cast<uint32_t>(input_dims[2]);
    input_type.dimensions[3] = static_cast<uint32_t>(input_dims[3]);
    NNAdapter::Global().NNAdapterGraph_addOperand(
        handle, &input_type, &input_operand);
    input_node = graph->Add(input_name, input_operand);
  }

  // Filter operand
  CHECK(op_info->HasInputScale(filter_scale_name, true));
  auto filter_scale = op_info->GetInputScale(filter_scale_name, true);
  bool is_perchannel_filter_scales = isPerChannelScales(filter_scale);
  NNAdapterOperand* filter_operand;
  NNAdapterOperandType filter_type;
  filter_type.dimensionCount = filter_dims.size();
  filter_type.dimensions[0] = static_cast<uint32_t>(filter_dims[0]);
  filter_type.dimensions[1] = static_cast<uint32_t>(filter_dims[1]);
  filter_type.dimensions[2] = static_cast<uint32_t>(filter_dims[2]);
  filter_type.dimensions[3] = static_cast<uint32_t>(filter_dims[3]);
  if (is_perchannel_filter_scales) {
    // Per channel
    filter_type.precision = NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL;
    filter_type.symmPerChannelParams.scales = &filter_scale[0];
    filter_type.symmPerChannelParams.scaleCount = filter_scale.size();
    filter_type.symmPerChannelParams.channelDim = 0;
  } else {
    // Per layer
    filter_type.precision = NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER;
    filter_type.symmPerLayerParams.scale = filter_scale[0];
  }
  NNAdapter::Global().NNAdapterGraph_addOperand(
      handle, &filter_type, &filter_operand);
  NNAdapter::Global().NNAdapterGraph_setOperand(
      filter_operand, filter->raw_data(), filter->memory_size());
  auto filter_node = graph->Add(filter_name, filter_operand);

  // Paddings, strides and dilations operands
  NNAdapterOperandType int32_type;
  int32_type.precision = NNADAPTER_INT32;
  int32_type.dimensionCount = 0;

  NNAdapterOperand* padding_width_left_operand;
  NNAdapter::Global().NNAdapterGraph_addOperand(
      handle, &int32_type, &padding_width_left_operand);
  NNAdapter::Global().NNAdapterGraph_setOperand(
      padding_width_left_operand, &paddings[0], sizeof(int32_t));
  auto padding_width_left_node = graph->Add(filter_name + "_padding_width_left",
                                            padding_width_left_operand);

  NNAdapterOperand* padding_width_right_operand;
  NNAdapter::Global().NNAdapterGraph_addOperand(
      handle, &int32_type, &padding_width_right_operand);
  NNAdapter::Global().NNAdapterGraph_setOperand(
      padding_width_right_operand, &paddings[1], sizeof(int32_t));
  auto padding_width_right_node = graph->Add(
      filter_name + "_padding_width_right", padding_width_right_operand);

  NNAdapterOperand* padding_height_top_operand;
  NNAdapter::Global().NNAdapterGraph_addOperand(
      handle, &int32_type, &padding_height_top_operand);
  NNAdapter::Global().NNAdapterGraph_setOperand(
      padding_height_top_operand, &paddings[2], sizeof(int32_t));
  auto padding_height_top_node = graph->Add(filter_name + "_padding_height_top",
                                            padding_height_top_operand);

  NNAdapterOperand* padding_height_bottom_operand;
  NNAdapter::Global().NNAdapterGraph_addOperand(
      handle, &int32_type, &padding_height_bottom_operand);
  NNAdapter::Global().NNAdapterGraph_setOperand(
      padding_height_bottom_operand, &paddings[3], sizeof(int32_t));
  auto padding_height_bottom_node = graph->Add(
      filter_name + "_padding_height_bottom", padding_height_bottom_operand);

  NNAdapterOperand* stride_width_operand;
  NNAdapter::Global().NNAdapterGraph_addOperand(
      handle, &int32_type, &stride_width_operand);
  NNAdapter::Global().NNAdapterGraph_setOperand(
      stride_width_operand, &strides[0], sizeof(int32_t));
  auto stride_width_node =
      graph->Add(filter_name + "_stride_width", stride_width_operand);

  NNAdapterOperand* stride_height_operand;
  NNAdapter::Global().NNAdapterGraph_addOperand(
      handle, &int32_type, &stride_height_operand);
  NNAdapter::Global().NNAdapterGraph_setOperand(
      stride_height_operand, &strides[1], sizeof(int32_t));
  auto stride_height_node =
      graph->Add(filter_name + "_stride_height", stride_height_operand);

  NNAdapterOperand* dilation_width_operand;
  NNAdapter::Global().NNAdapterGraph_addOperand(
      handle, &int32_type, &dilation_width_operand);
  NNAdapter::Global().NNAdapterGraph_setOperand(
      dilation_width_operand, &dilations[0], sizeof(int32_t));
  auto dilation_width_node =
      graph->Add(filter_name + "_dilation_width", dilation_width_operand);

  NNAdapterOperand* dilation_height_operand;
  NNAdapter::Global().NNAdapterGraph_addOperand(
      handle, &int32_type, &dilation_height_operand);
  NNAdapter::Global().NNAdapterGraph_setOperand(
      dilation_height_operand, &dilations[1], sizeof(int32_t));
  auto dilation_height_node =
      graph->Add(filter_name + "_dilation_height", dilation_height_operand);

  // Bias
  NNAdapterOperandType bias_type;
  std::vector<float> bias_scale(filter_scale.size());
  for (size_t i = 0; i < filter_scale.size(); i++) {
    bias_scale[i] = input_scale * filter_scale[i];
  }
  if (is_perchannel_filter_scales) {
    // Per channel
    bias_type.precision = NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL;
    bias_type.symmPerChannelParams.scales = &bias_scale[0];
    bias_type.symmPerChannelParams.scaleCount = bias_scale.size();
    bias_type.symmPerChannelParams.channelDim = 0;
  } else {
    // Per layer
    bias_type.precision = NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER;
    bias_type.symmPerLayerParams.scale = bias_scale[0];
  }
  bias_type.dimensionCount = 1;
  bias_type.dimensions[0] = static_cast<uint32_t>(output_channel_size);
  std::vector<int32_t> quant_bias_data(output_channel_size, 0);
  std::string bias_name = filter_name + "_dummy_bias";
  if (hasInput(op_info, scope, "Bias")) {
    bias_name = op_info->Input("Bias").front();
    auto bias = scope->FindMutableTensor(bias_name);
    auto bias_dims = bias->dims();
    CHECK((bias_dims.size() == 1 && bias_dims[0] == output_channel_size) ||
          (bias_dims.size() == 2 && bias_dims[0] == 1 &&
           bias_dims[1] == output_channel_size))
        << "The dimensions of bias only supports [C_out], [1, C_out]";
    auto* bias_data = bias->mutable_data<float>();
    quant(bias_data, output_channel_size, bias_scale, &quant_bias_data[0]);
  }
  NNAdapterOperand* bias_operand;
  NNAdapter::Global().NNAdapterGraph_addOperand(
      handle, &bias_type, &bias_operand);
  NNAdapter::Global().NNAdapterGraph_setOperand(
      bias_operand, &quant_bias_data[0], sizeof(int32_t));
  std::shared_ptr<Node> bias_node = graph->Add(bias_name, bias_operand);

  // Fuse code operand
  int32_t fuse_code_value = 0;
  if (act_type == "relu") {
    fuse_code_value = 1;
  } else if (act_type == "relu1") {
    fuse_code_value = 2;
  } else if (act_type == "relu6") {
    fuse_code_value = 3;
  } else if (!act_type.empty()) {
    fuse_code_value = 0;
    LOG(WARNING) << "Support act_type: " << act_type;
    return FAILED;
  }
  NNAdapterOperand* fuse_code_operand;
  NNAdapter::Global().NNAdapterGraph_addOperand(
      handle, &int32_type, &fuse_code_operand);
  NNAdapter::Global().NNAdapterGraph_setOperand(
      fuse_code_operand, &fuse_code_value, sizeof(int32_t));
  auto fuse_code_node =
      graph->Add(filter_name + "_fuse_code", fuse_code_operand);

  // Output operand
  NNAdapterOperandType output_type;
  output_type.precision = NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER;
  output_type.symmPerLayerParams.scale = output_scale;
  output_type.dimensionCount = output_dims.size();
  output_type.dimensions[0] = static_cast<uint32_t>(output_dims[0]);
  output_type.dimensions[1] = static_cast<uint32_t>(output_dims[1]);
  output_type.dimensions[2] = static_cast<uint32_t>(output_dims[2]);
  output_type.dimensions[3] = static_cast<uint32_t>(output_dims[3]);
  NNAdapterOperand* output_operand;
  NNAdapter::Global().NNAdapterGraph_addOperand(
      handle, &output_type, &output_operand);
  auto output_node = graph->Add(output_name, output_operand);

  // Conv2D operation
  std::vector<NNAdapterOperand*> input_operands = {
      input_node->data(),
      filter_node->data(),
      bias_node->data(),
      padding_width_left_node->data(),
      padding_width_right_node->data(),
      padding_height_top_node->data(),
      padding_height_bottom_node->data(),
      stride_width_node->data(),
      stride_height_node->data(),
      fuse_code_node->data(),
      dilation_width_node->data(),
      dilation_height_node->data()};
  std::vector<NNAdapterOperand*> output_operands = {output_node->data()};
  NNAdapterOperation* conv2d;
  NNAdapter::Global().NNAdapterGraph_addOperation(
      handle, NNADAPTER_CONV_2D, &conv2d);
  NNAdapter::Global().NNAdapterGraph_setOperation(conv2d,
                                                  input_operands.size(),
                                                  &input_operands[0],
                                                  output_operands.size(),
                                                  &output_operands[0]);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(conv2d,
                         kNNAdapter,
                         paddle::lite::subgraph::nnadapter::ConvConverter);
REGISTER_SUBGRAPH_BRIDGE(depthwise_conv2d,
                         kNNAdapter,
                         paddle::lite::subgraph::nnadapter::ConvConverter);
