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

#include "optimizer/fuse_conv2d_batch_norm_into_conv2d.h"
#include <algorithm>
#include <map>
#include <vector>
#include "optimizer/pattern_matcher.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {

class Conv2DBatchNormFuser : public PatternMatcher {
 public:
  explicit Conv2DBatchNormFuser(NNAdapterOperationType conv2d_type,
                                NNAdapterOperationType batch_norm_type)
      : conv2d_type_(conv2d_type), batch_norm_type_(batch_norm_type) {}
  void BuildPattern() override;
  bool HandleMatchedResults(core::Model* model,
                            const std::map<std::string, Node*>& nodes) override;

 private:
  NNAdapterOperationType conv2d_type_{NNADAPTER_CONV_2D};
  NNAdapterOperationType batch_norm_type_{NNADAPTER_BATCH_NORMALIZATION};
};

void Conv2DBatchNormFuser::BuildPattern() {
  // Operation patterns
  auto conv2d_pattern = CreatePattern("conv2d", conv2d_type_);
  auto batch_norm_pattern =
      CreatePattern("batch_norm", batch_norm_type_)->IsIntermediate();
  // Operand patterns
  auto conv2d_input_pattern =
      CreatePattern("conv2d_input")->IsOperationInputOperand(conv2d_type_, 0);
  auto conv2d_filter_pattern = CreatePattern("conv2d_filter")
                                   ->IsOperationInputOperand(conv2d_type_, 1)
                                   ->IsConstantOperand();
  auto conv2d_bias_pattern = CreatePattern("conv2d_bias")
                                 ->IsOperationInputOperand(conv2d_type_, 2)
                                 ->IsConstantOperand();
  auto conv2d_output_pattern = CreatePattern("conv2d_output")
                                   ->IsOperationOutputOperand(conv2d_type_, 0)
                                   ->IsIntermediate();
  auto batch_norm_scale_pattern =
      CreatePattern("batch_norm_scale")
          ->IsOperationInputOperand(batch_norm_type_, 1)
          ->IsConstantOperand()
          ->IsIntermediate();
  auto batch_norm_bias_pattern =
      CreatePattern("batch_norm_bias")
          ->IsOperationInputOperand(batch_norm_type_, 2)
          ->IsConstantOperand()
          ->IsIntermediate();
  auto batch_norm_mean_pattern =
      CreatePattern("batch_norm_mean")
          ->IsOperationInputOperand(batch_norm_type_, 3)
          ->IsConstantOperand()
          ->IsIntermediate();
  auto batch_norm_variance_pattern =
      CreatePattern("batch_norm_variance")
          ->IsOperationInputOperand(batch_norm_type_, 4)
          ->IsConstantOperand()
          ->IsIntermediate();
  auto batch_norm_epsilon_pattern =
      CreatePattern("batch_norm_epsilon")
          ->IsOperationInputOperand(batch_norm_type_, 5)
          ->IsConstantOperand()
          ->IsIntermediate();
  auto batch_norm_output_pattern =
      CreatePattern("batch_norm_output")
          ->IsOperationOutputOperand(batch_norm_type_, 0);
  // Create the topological connections for the above patterns
  std::vector<Pattern*> conv2d_input_patterns{
      conv2d_input_pattern, conv2d_filter_pattern, conv2d_bias_pattern};
  std::vector<Pattern*> batch_norm_input_patterns{conv2d_output_pattern,
                                                  batch_norm_scale_pattern,
                                                  batch_norm_bias_pattern,
                                                  batch_norm_mean_pattern,
                                                  batch_norm_variance_pattern,
                                                  batch_norm_epsilon_pattern};
  conv2d_input_patterns >> *conv2d_pattern >> *conv2d_output_pattern;
  batch_norm_input_patterns >> *batch_norm_pattern >>
      *batch_norm_output_pattern;
}

bool Conv2DBatchNormFuser::HandleMatchedResults(
    core::Model* model, const std::map<std::string, Node*>& nodes) {
  // Get the operands and operations from the matched subgraph nodes.
  auto conv2d_operation = nodes.at("conv2d")->operation;
  auto conv2d_input_operand = conv2d_operation->input_operands[0];
  auto conv2d_output_operand = conv2d_operation->output_operands[0];
  auto conv2d_filter_operand = conv2d_operation->input_operands[1];
  auto conv2d_bias_operand = conv2d_operation->input_operands[2];
  auto conv2d_group =
      *reinterpret_cast<int32_t*>(conv2d_operation->input_operands[6]->buffer);
  auto conv2d_output_channel_size =
      conv2d_filter_operand->type.dimensions.data[0];
  int32_t conv2d_filter_outer_size = 1;
  auto conv2d_filter_inner_size =
      ProductionOfDimensions(conv2d_filter_operand->type.dimensions.data,
                             conv2d_filter_operand->type.dimensions.count) /
      conv2d_output_channel_size;
  if (conv2d_type_ == NNADAPTER_CONV_2D_TRANSPOSE) {
    conv2d_output_channel_size =
        conv2d_filter_operand->type.dimensions.data[1] * conv2d_group;
    conv2d_filter_outer_size =
        conv2d_filter_operand->type.dimensions.data[0] / conv2d_group;
    conv2d_filter_inner_size = conv2d_filter_operand->type.dimensions.data[2] *
                               conv2d_filter_operand->type.dimensions.data[3];
  }
  auto batch_norm_operation = nodes.at("batch_norm")->operation;
  auto batch_norm_scale_data =
      reinterpret_cast<float*>(batch_norm_operation->input_operands[1]->buffer);
  auto batch_norm_bias_data =
      reinterpret_cast<float*>(batch_norm_operation->input_operands[2]->buffer);
  auto batch_norm_mean_data =
      reinterpret_cast<float*>(batch_norm_operation->input_operands[3]->buffer);
  auto batch_norm_variance_data =
      reinterpret_cast<float*>(batch_norm_operation->input_operands[4]->buffer);
  auto batch_norm_epsilon = *reinterpret_cast<float*>(
      batch_norm_operation->input_operands[5]->buffer);
  // The formula for BATCH_NORMALIZATION: output = scale * (input - mean) /
  // sqrt(variance + epsilon) + bias
  // Equivalent to: output = alpha * input + beta, where alpha = scale /
  // sqrt(variance + epsilon), beta = -scale * mean / sqrt(variance + epsilon) +
  // bias
  std::vector<float> batch_norm_alpha(conv2d_output_channel_size),
      batch_norm_beta(conv2d_output_channel_size);
  for (uint32_t i = 0; i < conv2d_output_channel_size; i++) {
    double coeff = batch_norm_scale_data[i] /
                   std::sqrt(batch_norm_variance_data[i] +
                             static_cast<double>(batch_norm_epsilon));
    batch_norm_alpha[i] = coeff;
    batch_norm_beta[i] =
        -batch_norm_mean_data[i] * coeff + batch_norm_bias_data[i];
  }
  if (IsInt8SymmPerLayerQuantType(conv2d_input_operand->type.precision) &&
      (IsInt8SymmPerLayerQuantType(conv2d_filter_operand->type.precision) ||
       IsInt8SymmPerChannelQuantType(conv2d_filter_operand->type.precision)) &&
      IsInt8SymmPerLayerQuantType(conv2d_output_operand->type.precision)) {
    auto conv2d_filter_data =
        reinterpret_cast<int8_t*>(conv2d_filter_operand->buffer);
    auto conv2d_bias_data =
        reinterpret_cast<int32_t*>(conv2d_bias_operand->buffer);
    std::vector<float> dequantized_conv2d_bias(conv2d_output_channel_size);
    if (IsInt8SymmPerLayerQuantType(conv2d_filter_operand->type.precision)) {
      NNADAPTER_CHECK(
          IsInt32SymmPerLayerQuantType(conv2d_bias_operand->type.precision));
      NNADAPTER_CHECK(DequantizeData<int32_t>(
          conv2d_bias_data,
          &conv2d_output_channel_size,
          1,
          &conv2d_bias_operand->type.symm_per_layer_params.scale,
          NULL,
          -1,
          -2147483647,
          2147483647,
          dequantized_conv2d_bias.data()));
      auto filter_scales = reinterpret_cast<float*>(
          malloc(conv2d_output_channel_size * sizeof(float)));
      NNADAPTER_CHECK(filter_scales) << "Failed to allocate the scale buffer "
                                        "for a symm per-channel quant type!";
      auto bias_scales = reinterpret_cast<float*>(
          malloc(conv2d_output_channel_size * sizeof(float)));
      NNADAPTER_CHECK(bias_scales) << "Failed to allocate the scale buffer for "
                                      "a symm per-channel quant type!";
      for (int64_t i = 0; i < conv2d_output_channel_size; i++) {
        filter_scales[i] =
            conv2d_filter_operand->type.symm_per_layer_params.scale;
        bias_scales[i] = conv2d_bias_operand->type.symm_per_layer_params.scale;
      }
      conv2d_filter_operand->type.symm_per_channel_params.scales =
          filter_scales;
      conv2d_filter_operand->type.symm_per_channel_params.scale_count =
          conv2d_output_channel_size;
      conv2d_filter_operand->type.symm_per_channel_params.channel_dim =
          conv2d_type_ == NNADAPTER_CONV_2D_TRANSPOSE ? 1 : 0;
      conv2d_filter_operand->type.precision =
          NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL;
      conv2d_bias_operand->type.symm_per_channel_params.scales = bias_scales;
      conv2d_bias_operand->type.symm_per_channel_params.scale_count =
          conv2d_output_channel_size;
      conv2d_bias_operand->type.symm_per_channel_params.channel_dim = 0;
      conv2d_bias_operand->type.precision =
          NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL;
    } else if (IsInt8SymmPerChannelQuantType(
                   conv2d_filter_operand->type.precision)) {
      NNADAPTER_CHECK(
          IsInt32SymmPerChannelQuantType(conv2d_bias_operand->type.precision));
      NNADAPTER_CHECK(DequantizeData<int32_t>(
          conv2d_bias_data,
          &conv2d_output_channel_size,
          1,
          conv2d_bias_operand->type.symm_per_channel_params.scales,
          NULL,
          conv2d_bias_operand->type.symm_per_channel_params.channel_dim,
          -2147483647,
          2147483647,
          dequantized_conv2d_bias.data()));
    }
    for (int64_t i = 0; i < conv2d_output_channel_size; i++) {
      conv2d_filter_operand->type.symm_per_channel_params.scales[i] *=
          fabsf(batch_norm_alpha[i]);
      conv2d_bias_operand->type.symm_per_channel_params.scales[i] *=
          fabsf(batch_norm_alpha[i]);
      dequantized_conv2d_bias[i] =
          batch_norm_alpha[i] * dequantized_conv2d_bias[i] + batch_norm_beta[i];
    }
    for (int64_t i = 0; i < conv2d_filter_outer_size; i++) {
      for (int64_t j = 0; j < conv2d_output_channel_size; j++) {
        if (batch_norm_alpha[j] >= 0.f) continue;
        for (int64_t k = 0; k < conv2d_filter_inner_size; k++) {
          conv2d_filter_data[i * conv2d_output_channel_size *
                                 conv2d_filter_inner_size +
                             j * conv2d_filter_inner_size + k] *= -1;
        }
      }
    }
    NNADAPTER_CHECK(QuantizeData<int32_t>(
        dequantized_conv2d_bias.data(),
        &conv2d_output_channel_size,
        1,
        conv2d_bias_operand->type.symm_per_channel_params.scales,
        NULL,
        conv2d_bias_operand->type.symm_per_channel_params.channel_dim,
        -2147483647,
        2147483647,
        conv2d_bias_data));
  } else {
    NNADAPTER_CHECK_EQ(conv2d_input_operand->type.precision, NNADAPTER_FLOAT32);
    NNADAPTER_CHECK_EQ(conv2d_filter_operand->type.precision,
                       NNADAPTER_FLOAT32);
    NNADAPTER_CHECK_EQ(conv2d_output_operand->type.precision,
                       NNADAPTER_FLOAT32);
    auto conv2d_filter_data =
        reinterpret_cast<float*>(conv2d_filter_operand->buffer);
    auto conv2d_bias_data =
        reinterpret_cast<float*>(conv2d_bias_operand->buffer);
    for (int64_t i = 0; i < conv2d_output_channel_size; i++) {
      conv2d_bias_data[i] =
          batch_norm_alpha[i] * conv2d_bias_data[i] + batch_norm_beta[i];
    }
    for (int64_t i = 0; i < conv2d_filter_outer_size; i++) {
      for (int64_t j = 0; j < conv2d_output_channel_size; j++) {
        for (int64_t k = 0; k < conv2d_filter_inner_size; k++) {
          conv2d_filter_data[i * conv2d_output_channel_size *
                                 conv2d_filter_inner_size +
                             j * conv2d_filter_inner_size + k] *=
              batch_norm_alpha[j];
        }
      }
    }
  }
  // Replace the output operand the of NNADAPTER_CONV_2D with the output operand
  // of NNADAPTER_BATCH_NORMALIZATION
  conv2d_operation->output_operands[0] =
      batch_norm_operation->output_operands[0];
  // The matched intermediate operands and operations will be deleted only when
  // it returns true.
  return true;
}

NNADAPTER_EXPORT void FuseConv2DBatchNormIntoConv2D(core::Model* model) {
  for (auto conv2d_type : {NNADAPTER_CONV_2D, NNADAPTER_CONV_2D_TRANSPOSE}) {
    for (auto batch_norm_type : {NNADAPTER_BATCH_NORMALIZATION}) {
      NNADAPTER_VLOG(5) << "Apply Conv2DBatchNormFuser for conv2d_type:"
                        << OperationTypeToString(conv2d_type)
                        << " batch_norm_type:"
                        << OperationTypeToString(batch_norm_type);
      bool stop;
      do {
        Conv2DBatchNormFuser conv2d_batch_norm_fuser(conv2d_type,
                                                     batch_norm_type);
        stop = conv2d_batch_norm_fuser.Apply(model) == 0;
      } while (!stop);
    }
  }
}

}  // namespace nnadapter
