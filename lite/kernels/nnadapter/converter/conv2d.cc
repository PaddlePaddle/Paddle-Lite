// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/nnadapter/converter/converter.h"
#include "lite/operators/conv_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace nnadapter {

int ConvertConv2D(Converter* converter, OpInfo* op, Scope* scope) {
  // Extract the inputs, outputs and attributes
  auto op_type = op->Type();
  auto input_name = op->Input("Input").front();
  auto input_scale_name = "Input0_scale";
  std::vector<float> input_scales;
  if (op->HasInputScale(input_scale_name, true)) {
    input_scales = op->GetInputScale(input_scale_name, true);
  }
  auto filter_name = op->Input("Filter").front();
  auto filter_scale_name = "Filter0_scale";
  auto filter_tensor = scope->FindMutableTensor(filter_name);
  CHECK(filter_tensor->persistable());
  std::vector<float> filter_scales;
  if (op->HasInputScale(filter_scale_name, true)) {
    filter_scales = op->GetInputScale(filter_scale_name, true);
    if (!IsValidSymmPerChannelQuantParams(filter_scales)) {
      filter_scales = {filter_scales[0]};
    }
  }
  auto filter_precison = filter_tensor->precision();
  auto filter_dims = filter_tensor->dims();
  auto output_channel_size = filter_dims[0];
  auto output_name = op->Output("Output").front();
  auto output_scale_name = "Output0_scale";
  std::vector<float> output_scales;
  if (op->HasOutputScale(output_scale_name, true)) {
    output_scales = op->GetOutputScale(output_scale_name, true);
  }
  std::vector<int> strides = op->GetAttr<std::vector<int>>("strides");
  std::vector<int> paddings = op->GetAttr<std::vector<int>>("paddings");
  auto groups = op->GetAttr<int>("groups");
  std::vector<int> dilations = op->GetAttr<std::vector<int>>("dilations");
  CHECK_EQ(dilations.size(), 2L);
  bool with_act = op->HasAttr("with_act") && op->GetAttr<bool>("with_act");
  std::string act_type = with_act ? op->GetAttr<std::string>("act_type") : "";
  auto fuse_relu = op->HasAttr("fuse_relu") && op->GetAttr<bool>("fuse_relu");
  if (fuse_relu) {
    CHECK(!with_act || (with_act && act_type == "relu"))
        << "There is a conflict between the attribute 'fuse_relu' and "
           "'with_act'.";
    with_act = true;
    act_type = "relu";
  }
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
  if (op->HasAttr("padding_algorithm")) {
    padding_algorithm = op->GetAttr<std::string>("padding_algorithm");
  }
  VLOG(5) << "padding_algorithm:" << padding_algorithm;
  // Check depthwise mode
  bool is_depthwise_mode = op_type == "depthwise_conv2d";
  VLOG(5) << "depthwise mode(" << is_depthwise_mode << ").";
  // Check quantization mode
  bool is_quant_mode = false;
  if (filter_precison == PRECISION(kInt8)) {
    CHECK(IsValidSymmQuantParams(filter_scales))
        << "Missing the quant params '" << filter_scale_name
        << "' for the input '" << filter_name << "'";
    CHECK(IsValidSymmPerLayerQuantParams(input_scales))
        << "Missing the quant params '" << input_scale_name
        << "' for the input '" << input_name << "'";
    CHECK(IsValidSymmPerLayerQuantParams(output_scales))
        << "Missing the quant params '" << output_scale_name
        << "' for the output '" << output_name << "'";
    is_quant_mode = true;
  }

  // Convert to NNAdapter operands and operation
  // Input operand
  auto input_operand =
      converter->AddInputOperand(scope, input_name, {}, input_scales);
  CHECK(input_operand);
  auto input_type = converter->GetOperandType(input_operand);
  // Check depthwise mode according to the dimensions
  auto input_channel_size = input_type->dimensions.data[1];
  // Should not support dynamic shape for input channel size
  CHECK(input_channel_size != NNADAPTER_UNKNOWN);
  is_depthwise_mode |= (groups != 1 && input_channel_size == groups &&
                        output_channel_size % input_channel_size == 0);
  VLOG(5) << "Update depthwise mode(" << is_depthwise_mode << ").";
  // Filter operand
  NNAdapterOperand* filter_operand = nullptr;
  std::vector<float> bias_scales;
  if (is_quant_mode) {
    CHECK(IsNNInt8SymmPerLayerQuantType(*input_type));
    std::vector<float> quant_scales;
    CHECK(GetNNSymmQuantParams(*input_type, &quant_scales));
    CHECK(IsSameSymmQuantParams(input_scales, quant_scales));
    filter_operand =
        converter->AddConstantOperand(*filter_tensor, {}, false, filter_scales);
    bias_scales.resize(filter_scales.size());
    for (size_t i = 0; i < filter_scales.size(); i++) {
      bias_scales[i] = input_scales[0] * filter_scales[i];
    }
  } else {
    CHECK(input_type->precision ==
          ConvertPrecisionTypeToNNPrecisionCode(filter_precison));
    filter_operand = converter->AddConstantOperand(*filter_tensor);
  }
  // Bias operand
  NNAdapterOperand* bias_operand = nullptr;
  if (HasInput(op, scope, "Bias")) {
    auto bias_name = op->Input("Bias").front();
    bias_operand = converter->GetMappedOperand(bias_name);
    if (!bias_operand) {
      auto bias_tensor = scope->FindMutableTensor(bias_name);
      CHECK(bias_tensor->persistable());
      auto bias_precison = bias_tensor->precision();
      auto bias_dims = bias_tensor->dims();
      if (bias_dims.production() != output_channel_size) {
        LOG(FATAL)
            << "Only supports bias_dims.production() == output_channel_size !";
        return UNSUPPORTED_FEATURE;
      }
      if (is_quant_mode) {
        CHECK(bias_tensor->precision() == PRECISION(kFloat));
        auto bias_data = bias_tensor->mutable_data<float>();
        std::vector<int32_t> quantized_bias_data(output_channel_size, 0);
        SymmQuantizeData(bias_data,
                         output_channel_size,
                         bias_scales,
                         &quantized_bias_data[0]);
        bias_operand = converter->AddConstantOperand(
            quantized_bias_data, DDim({output_channel_size}), bias_scales);
      } else {
        CHECK(input_type->precision ==
              ConvertPrecisionTypeToNNPrecisionCode(bias_precison));
        bias_operand = converter->AddConstantOperand(
            *bias_tensor, DDim({output_channel_size}));
      }
    } else {
      auto bias_type = converter->GetOperandType(bias_operand);
      // Check if we can use the bias_operand directly
      if (is_quant_mode) {
        CHECK(IsNNInt32SymmQuantType(*bias_type));
        std::vector<float> quant_scales;
        CHECK(GetNNSymmQuantParams(*bias_type, &quant_scales));
        CHECK(IsSameSymmQuantParams(bias_scales, quant_scales));
      } else {
        CHECK(bias_type->precision == input_type->precision);
      }
    }
  } else {
    // Add dummy zero bias operand
    // Use int32 as the data type of bias if it is a quantized type
    std::vector<int8_t> zeros(
        output_channel_size *
            (is_quant_mode ? sizeof(int32_t)
                           : GetNNOperandPrecisionDataLength(*input_type)),
        0);
    bias_operand = converter->AddConstantOperand(
        reinterpret_cast<void*>(zeros.data()),
        DDim({output_channel_size}),
        is_quant_mode ? NNADAPTER_INT32 : input_type->precision,
        true,
        bias_scales);
  }
  // Auto_pad operand
  auto auto_pad_operand = converter->AddConstantOperand(static_cast<int32_t>(
      ConvertPaddingAlgorithmToNNAutoPadCode(padding_algorithm)));
  // Pads operand(optional)
  auto pads_operand = converter->AddConstantOperand(paddings);
  // Strides operand
  auto strides_operand = converter->AddConstantOperand(strides);
  // Group operand
  auto group_operand = converter->AddConstantOperand(groups);
  // Dilations operand
  auto dilations_operand = converter->AddConstantOperand(dilations);
  // Fuse code operand
  int32_t fuse_code_value = NNADAPTER_FUSED_NONE;
  if (act_type == "relu") {
    fuse_code_value = NNADAPTER_FUSED_RELU;
    act_type = "";
  } else if (act_type == "relu1") {
    fuse_code_value = NNADAPTER_FUSED_RELU1;
    act_type = "";
  } else if (act_type == "relu6") {
    fuse_code_value = NNADAPTER_FUSED_RELU6;
    act_type = "";
  }
  auto fuse_code_operand = converter->AddConstantOperand(fuse_code_value);
  // Output operand
  auto output_operand = converter->AddOutputOperand(output_name, output_scales);
  // Conv2D operation
  converter->AddOperation(NNADAPTER_CONV_2D,
                          {input_operand,
                           filter_operand,
                           bias_operand,
                           auto_pad_operand,
                           pads_operand,
                           strides_operand,
                           group_operand,
                           dilations_operand,
                           fuse_code_operand},
                          {output_operand});
  // Unpack the fused activations
  converter->UnpackFusedActivations(
      output_operand, act_type, op, scope, output_name, output_scales);
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
