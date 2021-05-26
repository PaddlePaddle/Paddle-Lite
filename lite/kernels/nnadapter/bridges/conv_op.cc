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
#include "lite/kernels/nnadapter/bridges/converter.h"
#include "lite/kernels/nnadapter/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace nnadapter {

int ConvConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto converter = static_cast<Converter*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "Converting " << op_type << " ...";

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
  auto filter_channel_size = filter_dims[1];
  CHECK_EQ(input_dims.size(), 4L);
  CHECK_EQ(output_dims.size(), 4L);
  CHECK_EQ(filter_dims.size(), 4L);
  CHECK_EQ(output_dims[0], batch_size);
  CHECK_EQ(output_dims[1], output_channel_size);
  auto strides = op_info->GetAttr<std::vector<int>>("strides");
  std::vector<int> paddings = op_info->GetAttr<std::vector<int>>("paddings");
  auto groups = op_info->GetAttr<int>("groups");
  std::vector<int> dilations = op_info->GetAttr<std::vector<int>>("dilations");
  CHECK_EQ(dilations.size(), 2L);
  bool with_act =
      op_info->HasAttr("with_act") && op_info->GetAttr<bool>("with_act");
  std::string act_type =
      with_act ? op_info->GetAttr<std::string>("act_type") : "";
  float leaky_relu_alpha = act_type == "leaky_relu"
                               ? op_info->GetAttr<float>("leaky_relu_alpha")
                               : 0.f;
  // Calculate paddings and strides
  CHECK_EQ(strides.size(), 2L);
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
  bool is_depthwise_mode =
      (input_channel_size == groups && output_channel_size == groups &&
       filter_channel_size == 1);
  VLOG(5) << "depthwise mode(" << is_depthwise_mode << ").";

  // Input operand
  CHECK(op_info->HasInputScale(input_scale_name, true));
  auto input_scale = op_info->GetInputScale(input_scale_name, true)[0];
  NNAdapterOperand* input_operand = nullptr;
  if (converter->HasOperand(input_name)) {
    input_operand = converter->GetOperand(input_name);
  } else {
    NNAdapterOperandType input_type;
    memset(&input_type, 0, sizeof(NNAdapterOperandType));
    input_type.precision = NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER;
    input_type.symm_per_layer_params.scale = input_scale;
    ConvertDimensions(
        input_dims, input_type.dimensions, &input_type.dimension_count);
    input_operand = converter->AddOperand(&input_type, input_name);
  }

  // Filter operand
  CHECK(op_info->HasInputScale(filter_scale_name, true));
  auto filter_scale = op_info->GetInputScale(filter_scale_name, true);
  bool is_per_channel = IsPerChannelScales(filter_scale);
  VLOG(5) << "is_per_channel: " << is_per_channel;
  NNAdapterOperandType filter_type;
  memset(&filter_type, 0, sizeof(NNAdapterOperandType));
  ConvertDimensions(
      filter_dims, filter_type.dimensions, &filter_type.dimension_count);
  if (is_per_channel) {
    // Per channel
    filter_type.precision = NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL;
    filter_type.symm_per_channel_params.scales = &filter_scale[0];
    filter_type.symm_per_channel_params.scale_count = filter_scale.size();
    filter_type.symm_per_channel_params.channel_dim = 0;
  } else {
    // Per layer
    filter_type.precision = NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER;
    filter_type.symm_per_layer_params.scale = filter_scale[0];
  }
  auto filter_operand = converter->AddOperand(&filter_type, filter_name);
  converter->SetOperandReferenceTo(
      filter_operand, filter->raw_data(), filter->memory_size());

  // Paddings, strides, dilations and group operands
  NNAdapterOperandType int32_type;
  memset(&int32_type, 0, sizeof(NNAdapterOperandType));
  int32_type.precision = NNADAPTER_INT32;
  int32_type.dimension_count = 0;

  auto padding_width_left_operand = converter->AddOperand(&int32_type);
  converter->SetOperandCopyFrom(
      padding_width_left_operand, &paddings[0], sizeof(int32_t));

  auto padding_width_right_operand = converter->AddOperand(&int32_type);
  converter->SetOperandCopyFrom(
      padding_width_right_operand, &paddings[1], sizeof(int32_t));

  auto padding_height_top_operand = converter->AddOperand(&int32_type);
  converter->SetOperandCopyFrom(
      padding_height_top_operand, &paddings[2], sizeof(int32_t));

  auto padding_height_bottom_operand = converter->AddOperand(&int32_type);
  converter->SetOperandCopyFrom(
      padding_height_bottom_operand, &paddings[3], sizeof(int32_t));

  auto stride_width_operand = converter->AddOperand(&int32_type);
  converter->SetOperandCopyFrom(
      stride_width_operand, &strides[0], sizeof(int32_t));

  auto stride_height_operand = converter->AddOperand(&int32_type);
  converter->SetOperandCopyFrom(
      stride_height_operand, &strides[1], sizeof(int32_t));

  auto dilation_width_operand = converter->AddOperand(&int32_type);
  converter->SetOperandCopyFrom(
      dilation_width_operand, &dilations[0], sizeof(int32_t));

  auto dilation_height_operand = converter->AddOperand(&int32_type);
  converter->SetOperandCopyFrom(
      dilation_height_operand, &dilations[1], sizeof(int32_t));

  auto group_operand = converter->AddOperand(&int32_type);
  converter->SetOperandCopyFrom(group_operand, &groups, sizeof(int32_t));

  // Bias
  NNAdapterOperandType bias_type;
  memset(&bias_type, 0, sizeof(NNAdapterOperandType));
  std::vector<float> bias_scale(filter_scale.size());
  for (size_t i = 0; i < filter_scale.size(); i++) {
    bias_scale[i] = input_scale * filter_scale[i];
  }
  if (is_per_channel) {
    // Per channel
    bias_type.precision = NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_CHANNEL;
    bias_type.symm_per_channel_params.scales = &bias_scale[0];
    bias_type.symm_per_channel_params.scale_count = bias_scale.size();
    bias_type.symm_per_channel_params.channel_dim = 0;
  } else {
    // Per layer
    bias_type.precision = NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_LAYER;
    bias_type.symm_per_layer_params.scale = bias_scale[0];
  }
  bias_type.dimension_count = 1;
  bias_type.dimensions[0] = static_cast<int32_t>(output_channel_size);
  std::vector<int32_t> quant_bias_data(output_channel_size, 0);
  std::string bias_name = output_name + "_dummy_bias";
  if (HasInput(op_info, scope, "Bias")) {
    bias_name = op_info->Input("Bias").front();
    auto bias = scope->FindMutableTensor(bias_name);
    auto bias_dims = bias->dims();
    CHECK((bias_dims.size() == 1 && bias_dims[0] == output_channel_size) ||
          (bias_dims.size() == 2 && bias_dims[0] == 1 &&
           bias_dims[1] == output_channel_size))
        << "The dimensions of bias only supports [C_out], [1, C_out]";
    auto* bias_data = bias->mutable_data<float>();
    Quantize(bias_data, output_channel_size, bias_scale, &quant_bias_data[0]);
  }
  auto bias_operand = converter->AddOperand(&bias_type, bias_name);
  converter->SetOperandCopyFrom(bias_operand,
                                &quant_bias_data[0],
                                sizeof(int32_t) * quant_bias_data.size());
  // Fuse code operand
  int32_t fuse_code_value = NNADAPTER_FUSED_NONE;
  if (act_type == "relu") {
    fuse_code_value = 1;
  } else if (act_type == "relu1") {
    fuse_code_value = 2;
  } else if (act_type == "relu6") {
    fuse_code_value = 3;
  } else if (!act_type.empty()) {
    LOG(WARNING) << "Unsupported activation type: " << act_type;
    return FAILED;
  }
  auto fuse_code_operand = converter->AddOperand(&int32_type);
  converter->SetOperandCopyFrom(
      fuse_code_operand, &fuse_code_value, sizeof(int32_t));
  // Output operand
  CHECK(op_info->HasOutputScale(output_scale_name, true));
  auto output_scale = op_info->GetOutputScale(output_scale_name, true)[0];
  NNAdapterOperandType output_type;
  memset(&output_type, 0, sizeof(NNAdapterOperandType));
  output_type.precision = NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER;
  output_type.symm_per_layer_params.scale = output_scale;
  ConvertDimensions(
      output_dims, output_type.dimensions, &output_type.dimension_count);
  auto output_operand = converter->AddOperand(&output_type, output_name);

  // Conv2D operation
  std::vector<NNAdapterOperand*> input_operands = {
      input_operand,
      filter_operand,
      bias_operand,
      padding_width_left_operand,
      padding_width_right_operand,
      padding_height_top_operand,
      padding_height_bottom_operand,
      stride_width_operand,
      stride_height_operand,
      group_operand,
      fuse_code_operand,
      dilation_width_operand,
      dilation_height_operand};
  std::vector<NNAdapterOperand*> output_operands = {output_operand};
  auto conv2d_operation = converter->AddOperation(NNADAPTER_CONV_2D);
  converter->SetOperation(conv2d_operation, &input_operands, &output_operands);
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
