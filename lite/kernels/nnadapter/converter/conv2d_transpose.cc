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

namespace paddle {
namespace lite {
namespace kernels {
namespace nnadapter {

int ConvertConv2dTranspose(Converter* converter, OpInfo* op, Scope* scope) {
  // Input operand
  auto input_name = op->Input("Input").front();
  auto input_scale_name = "Input0_scale";
  std::vector<float> input_scales =
      op->HasInputScale(input_scale_name, true)
          ? op->GetInputScale(input_scale_name, true)
          : std::vector<float>();
  auto input_operand =
      converter->AddInputOperand(scope, input_name, {}, input_scales);

  // Filter operand
  auto filter_name = op->Input("Filter").front();
  auto filter_scale_name = "Filter0_scale";
  std::vector<float> filter_scales =
      op->HasInputScale(filter_scale_name, true)
          ? op->GetInputScale(filter_scale_name, true)
          : std::vector<float>();
  auto filter_operand =
      converter->AddInputOperand(scope, filter_name, {}, filter_scales);

  // Bias operand
  NNAdapterOperand* bias_operand = nullptr;
  std::vector<float> bias_scales(filter_scales.size());
  if (!input_scales.empty()) {
    for (size_t i = 0; i < filter_scales.size(); i++) {
      bias_scales[i] = input_scales[0] * filter_scales[i];
    }
  }
  if (HasInput(op, scope, "Bias")) {
    auto bias_name = op->Input("Bias").front();
    bias_operand =
        converter->AddInputOperand(scope, bias_name, {}, bias_scales);
  } else {
    // Dummy bias
    int groups = op->GetAttr<int>("groups");
    int output_channel_size =
        scope->FindTensor(filter_name)->dims()[1] * groups;
    std::vector<float> bias(output_channel_size, 0.f);
    bias_operand = converter->AddConstantOperand(
        bias, DDim({output_channel_size}), bias_scales);
  }

  // Auto_pad operand
  std::string padding_algorithm =
      op->HasAttr("padding_algorithm")
          ? op->GetAttr<std::string>("padding_algorithm")
          : "";
  auto auto_pad_operand = converter->AddConstantOperand(static_cast<int32_t>(
      ConvertPaddingAlgorithmToNNAutoPadCode(padding_algorithm)));

  // Pads operand(optional)
  std::vector<int> paddings = op->GetAttr<std::vector<int>>("paddings");
  if (paddings.size() == 2UL) {
    paddings.insert(paddings.begin(), paddings[0]);
    paddings.insert(paddings.begin() + 2, paddings[2]);
  }
  auto pads_operand = converter->AddConstantOperand(paddings);

  // Strides operand
  std::vector<int> strides = op->GetAttr<std::vector<int>>("strides");
  auto strides_operand = converter->AddConstantOperand(strides);

  // Group operand
  int groups = op->GetAttr<int>("groups");
  auto group_operand = converter->AddConstantOperand(groups);

  // Dilations operand
  std::vector<int> dilations = op->GetAttr<std::vector<int>>("dilations");
  auto dilations_operand = converter->AddConstantOperand(dilations);

  // Output_padding operand
  std::vector<int> output_padding =
      op->HasAttr("output_padding")
          ? op->GetAttr<std::vector<int>>("output_padding")
          : std::vector<int>(2, 0);
  if (output_padding.size() == 0) {
    output_padding = std::vector<int>(2, 0);
  }
  auto output_padding_operand = converter->AddConstantOperand(output_padding);

  // Output_shape operand
  NNAdapterOperand* output_shape_operand = nullptr;
  if (op->HasAttr("output_size") &&
      !op->GetAttr<std::vector<int>>("output_size").empty()) {
    std::vector<int> output_size = op->GetAttr<std::vector<int>>("output_size");
    output_shape_operand = converter->AddConstantOperand(output_size);
  }

  // Fuse code operand
  bool with_act =
      op->HasAttr("with_act") ? op->GetAttr<bool>("with_act") : false;
  std::string act_type = with_act ? op->GetAttr<std::string>("act_type") : "";
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
  } else if (!act_type.empty()) {
    LOG(FATAL) << "Unsupported activation type: " << act_type;
  }
  LOG(INFO) << "--- with_act: " << with_act << ", act_type: " << act_type;
  auto fuse_code_operand = converter->AddConstantOperand(fuse_code_value);

  // Output operand
  auto output_name = op->Output("Output").front();
  auto output_scale_name = "Output0_scale";
  std::vector<float> output_scales =
      op->HasOutputScale(output_scale_name, true)
          ? op->GetOutputScale(output_scale_name, true)
          : std::vector<float>();
  auto output_operand = converter->AddOutputOperand(output_name, output_scales);

  // Conv2d_transpose operation
  converter->AddOperation(NNADAPTER_CONV_2D_TRANSPOSE,
                          {input_operand,
                           filter_operand,
                           bias_operand,
                           auto_pad_operand,
                           pads_operand,
                           strides_operand,
                           group_operand,
                           dilations_operand,
                           output_padding_operand,
                           output_shape_operand,
                           fuse_code_operand},
                          {output_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
