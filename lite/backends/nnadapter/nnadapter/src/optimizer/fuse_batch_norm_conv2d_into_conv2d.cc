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

#include "optimizer/fuse_batch_norm_conv2d_into_conv2d.h"
#include <algorithm>
#include <iostream>
#include <map>
#include <vector>
#include "optimizer/pattern_matcher.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {

class BatchNormConv2DFuser : public PatternMatcher {
 public:
  explicit BatchNormConv2DFuser(NNAdapterOperationType batch_norm_type,
                                NNAdapterOperationType conv2d_type)
      : batch_norm_type_(batch_norm_type), conv2d_type_(conv2d_type) {}
  void BuildPattern() override;
  bool HandleMatchedResults(core::Model* model,
                            const std::map<std::string, Node*>& nodes) override;

 private:
  NNAdapterOperationType batch_norm_type_{NNADAPTER_BATCH_NORMALIZATION};
  NNAdapterOperationType conv2d_type_{NNADAPTER_CONV_2D};
};

void BatchNormConv2DFuser::BuildPattern() {
  // Operation patterns
  auto batch_norm_pattern =
      CreatePattern("batch_norm", batch_norm_type_)->IsIntermediate();
  auto conv2d_pattern = CreatePattern("conv2d", conv2d_type_);
  // Operand patterns
  auto batch_norm_input_pattern =
      CreatePattern("batch_norm_input")
          ->IsOperationInputOperand(batch_norm_type_, 0);
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
          ->IsOperationOutputOperand(batch_norm_type_, 0)
          ->IsOperationInputOperand(conv2d_type_, 0)
          ->IsIntermediate();
  auto conv2d_filter_pattern = CreatePattern("conv2d_filter")
                                   ->IsOperationInputOperand(conv2d_type_, 1)
                                   ->IsConstantOperand();
  auto conv2d_bias_pattern = CreatePattern("conv2d_bias")
                                 ->IsOperationInputOperand(conv2d_type_, 2)
                                 ->IsConstantOperand();
  auto conv2d_output_pattern =
      CreatePattern("conv2d_output")->IsOperationOutputOperand(conv2d_type_, 0);
  // Create the topological connections for the above patterns
  std::vector<Pattern*> batch_norm_input_patterns{batch_norm_input_pattern,
                                                  batch_norm_scale_pattern,
                                                  batch_norm_bias_pattern,
                                                  batch_norm_mean_pattern,
                                                  batch_norm_variance_pattern,
                                                  batch_norm_epsilon_pattern};
  std::vector<Pattern*> conv2d_input_patterns{
      batch_norm_output_pattern, conv2d_filter_pattern, conv2d_bias_pattern};
  batch_norm_input_patterns >> *batch_norm_pattern >>
      *batch_norm_output_pattern;
  conv2d_input_patterns >> *conv2d_pattern >> *conv2d_output_pattern;
}

bool BatchNormConv2DFuser::HandleMatchedResults(
    core::Model* model, const std::map<std::string, Node*>& nodes) {
  // Get the operands and operations from the matched subgraph nodes.
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
  auto conv2d_operation = nodes.at("conv2d")->operation;
  auto conv2d_input_operand = conv2d_operation->input_operands[0];
  auto& conv2d_input_type = conv2d_input_operand->type;
  auto conv2d_output_operand = conv2d_operation->output_operands[0];
  auto& conv2d_output_type = conv2d_output_operand->type;
  auto conv2d_filter_operand = conv2d_operation->input_operands[1];
  auto& conv2d_filter_type = conv2d_filter_operand->type;
  auto conv2d_bias_operand = conv2d_operation->input_operands[2];
  auto conv2d_group =
      *reinterpret_cast<int32_t*>(conv2d_operation->input_operands[6]->buffer);
  auto conv2d_input_channel_size = conv2d_input_type.dimensions.data[1];
  NNADAPTER_CHECK_NE(conv2d_input_channel_size, NNADAPTER_UNKNOWN);
  auto conv2d_output_channel_size = conv2d_filter_type.dimensions.data[0];
  auto conv2d_input_channel_group = conv2d_input_channel_size / conv2d_group;
  auto conv2d_output_channel_group = conv2d_output_channel_size / conv2d_group;
  auto conv2d_filter_inner_size = conv2d_filter_type.dimensions.data[2] *
                                  conv2d_filter_type.dimensions.data[3];
  // The formula for BATCH_NORMALIZATION: output = scale * (input - mean) /
  // sqrt(variance + epsilon) + bias
  // Equivalent to: output = alpha * input + beta, where alpha = scale /
  // sqrt(variance + epsilon), beta = -scale * mean / sqrt(variance + epsilon) +
  // bias
  std::vector<double> batch_norm_alpha(conv2d_input_channel_size),
      batch_norm_beta(conv2d_input_channel_size);
  for (int64_t i = 0; i < conv2d_input_channel_size; i++) {
    double coeff = batch_norm_scale_data[i] /
                   std::sqrt(static_cast<double>(batch_norm_variance_data[i]) +
                             batch_norm_epsilon);
    batch_norm_alpha[i] = coeff;
    batch_norm_beta[i] =
        -batch_norm_mean_data[i] * coeff + batch_norm_bias_data[i];
  }
  if (IsInt8SymmPerLayerQuantType(conv2d_input_type.precision) &&
      (IsInt8SymmPerLayerQuantType(conv2d_filter_type.precision) ||
       IsInt8SymmPerChannelQuantType(conv2d_filter_type.precision)) &&
      IsInt8SymmPerLayerQuantType(conv2d_output_type.precision)) {
    // TODO(hong19860320) Add bn+conv2d fusion for the quantized conv2d
    return false;
  } else {
    NNADAPTER_CHECK_EQ(conv2d_input_type.precision, NNADAPTER_FLOAT32);
    NNADAPTER_CHECK_EQ(conv2d_filter_type.precision, NNADAPTER_FLOAT32);
    NNADAPTER_CHECK_EQ(conv2d_output_type.precision, NNADAPTER_FLOAT32);
    auto conv2d_filter_data =
        reinterpret_cast<float*>(conv2d_filter_operand->buffer);
    auto conv2d_bias_data =
        reinterpret_cast<float*>(conv2d_bias_operand->buffer);
    for (int64_t g = 0; g < conv2d_group; g++) {
      for (int64_t i = 0; i < conv2d_output_channel_group; i++) {
        float sum = 0.0f;
        for (int64_t j = 0; j < conv2d_input_channel_group; j++) {
          for (int64_t k = 0; k < conv2d_filter_inner_size; k++) {
            auto offset =
                g * conv2d_output_channel_group * conv2d_input_channel_group *
                    conv2d_filter_inner_size +
                i * conv2d_input_channel_group * conv2d_filter_inner_size +
                j * conv2d_filter_inner_size + k;
            auto value = conv2d_filter_data[offset];
            conv2d_filter_data[offset] =
                value * batch_norm_alpha[g * conv2d_input_channel_group + j];
            sum += value * batch_norm_beta[g * conv2d_input_channel_group + j];
          }
        }
        conv2d_bias_data[g * conv2d_output_channel_group + i] += sum;
      }
    }
  }
  // Replace the input operand the of NNADAPTER_CONV_2D with the input operand
  // of NNADAPTER_BATCH_NORMALIZATION
  conv2d_operation->input_operands[0] = batch_norm_operation->input_operands[0];
  // The matched intermediate operands and operations will be deleted only when
  // it returns true.
  return true;
}

NNADAPTER_EXPORT void FuseBatchNormConv2DIntoConv2D(core::Model* model) {
  for (auto batch_norm_type : {NNADAPTER_BATCH_NORMALIZATION}) {
    for (auto conv2d_type : {NNADAPTER_CONV_2D}) {
      NNADAPTER_VLOG(5) << "Apply BatchNormConv2DFuser for batch_norm_type:"
                        << OperationTypeToString(batch_norm_type)
                        << " conv2d_type:"
                        << OperationTypeToString(conv2d_type);
      bool stop;
      do {
        BatchNormConv2DFuser batch_norm_conv2d_fuser(batch_norm_type,
                                                     conv2d_type);
        stop = batch_norm_conv2d_fuser.Apply(model) == 0;
      } while (!stop);
    }
  }
}

}  // namespace nnadapter
