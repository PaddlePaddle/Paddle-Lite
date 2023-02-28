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

#include "optimizer/fuse_conv2d_add_into_conv2d.h"
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

class Conv2DAddFuser : public PatternMatcher {
 public:
  explicit Conv2DAddFuser(NNAdapterOperationType conv2d_type)
      : conv2d_type_(conv2d_type) {}
  void BuildPattern() override;
  bool HandleMatchedResults(core::Model* model,
                            const std::map<std::string, Node*>& nodes) override;

 private:
  NNAdapterOperationType conv2d_type_{NNADAPTER_CONV_2D};
};

void Conv2DAddFuser::BuildPattern() {
  // Operation patterns
  auto conv2d_pattern = CreatePattern("conv2d", conv2d_type_);
  auto add_pattern = CreatePattern("add", NNADAPTER_ADD)->IsIntermediate();
  // Operand patterns
  auto conv2d_input_pattern =
      CreatePattern("conv2d_input")->IsOperationInputOperand(conv2d_type_, 0);
  int conv2d_fuse_code_index = -1;
  if (conv2d_type_ == NNADAPTER_CONV_2D) {
    conv2d_fuse_code_index = 8;
  } else if (conv2d_type_ == NNADAPTER_CONV_2D_TRANSPOSE) {
    conv2d_fuse_code_index = 10;
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported operation type ("
                         << OperationTypeToString(conv2d_type_) << ") !";
  }
  auto conv2d_fuse_code_pattern =
      CreatePattern("conv2d_fuse_code")
          ->IsOperationInputOperand(conv2d_type_, conv2d_fuse_code_index)
          ->IsConstantOperand()
          ->MatchCondition([](const Node* node) -> bool {
            auto operand = node->operand;
            return operand &&
                   *reinterpret_cast<int32_t*>(operand->buffer) ==
                       NNADAPTER_FUSED_NONE;
          });
  auto conv2d_output_pattern = CreatePattern("conv2d_output")
                                   ->IsOperationOutputOperand(conv2d_type_, 0)
                                   ->IsOperationInputOperand(NNADAPTER_ADD, 0)
                                   ->IsIntermediate();
  auto add_input_pattern = CreatePattern("add_input")
                               ->IsOperationInputOperand(NNADAPTER_ADD, 1)
                               ->IsConstantOperand()
                               ->IsIntermediate();
  auto add_fuse_code_pattern = CreatePattern("add_fuse_code")
                                   ->IsOperationInputOperand(NNADAPTER_ADD, 2)
                                   ->IsConstantOperand()
                                   ->IsIntermediate();
  auto add_output_pattern =
      CreatePattern("add_output")->IsOperationOutputOperand(NNADAPTER_ADD, 0);
  // Create the topological connections for the above patterns
  std::vector<Pattern*> conv2d_input_patterns{conv2d_input_pattern,
                                              conv2d_fuse_code_pattern};
  std::vector<Pattern*> add_input_patterns{
      conv2d_output_pattern, add_input_pattern, add_fuse_code_pattern};
  conv2d_input_patterns >> *conv2d_pattern >> *conv2d_output_pattern;
  add_input_patterns >> *add_pattern >> *add_output_pattern;
}

bool Conv2DAddFuser::HandleMatchedResults(
    core::Model* model, const std::map<std::string, Node*>& nodes) {
  // Get the operands and operations from the matched subgraph nodes.
  auto conv2d_operation = nodes.at("conv2d")->operation;
  auto conv2d_fuse_code_operand = nodes.at("conv2d_fuse_code")->operand;
  auto conv2d_filter_operand = conv2d_operation->input_operands[1];
  auto conv2d_bias_operand = conv2d_operation->input_operands[2];
  auto conv2d_group =
      *reinterpret_cast<int32_t*>(conv2d_operation->input_operands[6]->buffer);
  auto conv2d_output_channel_size =
      conv2d_filter_operand->type.dimensions.data[0];
  if (conv2d_type_ == NNADAPTER_CONV_2D_TRANSPOSE) {
    conv2d_output_channel_size =
        conv2d_filter_operand->type.dimensions.data[1] * conv2d_group;
  }
  auto add_input_operand = nodes.at("add_input")->operand;
  auto add_fuse_code_operand = nodes.at("add_fuse_code")->operand;
  auto add_output_operand = nodes.at("add_output")->operand;
  if (ProductionOfDimensions(add_input_operand->type.dimensions.data,
                             add_input_operand->type.dimensions.count) !=
      conv2d_output_channel_size)
    return false;
  // Dequantize the bias operand of NNADAPTER_CONV_2D and the constant input
  // operand of NNADAPTER_ADD
  std::vector<float> dequantized_conv2d_bias(conv2d_output_channel_size);
  if (IsInt32SymmPerLayerQuantType(conv2d_bias_operand->type.precision)) {
    NNADAPTER_CHECK(DequantizeData<int32_t>(
        reinterpret_cast<int32_t*>(conv2d_bias_operand->buffer),
        &conv2d_output_channel_size,
        1,
        &conv2d_bias_operand->type.symm_per_layer_params.scale,
        NULL,
        -1,
        -2147483647,
        2147483647,
        dequantized_conv2d_bias.data()));
  } else if (IsInt32SymmPerChannelQuantType(
                 conv2d_bias_operand->type.precision)) {
    NNADAPTER_CHECK(DequantizeData<int32_t>(
        reinterpret_cast<int32_t*>(conv2d_bias_operand->buffer),
        &conv2d_output_channel_size,
        1,
        conv2d_bias_operand->type.symm_per_channel_params.scales,
        NULL,
        conv2d_bias_operand->type.symm_per_channel_params.channel_dim,
        -2147483647,
        2147483647,
        dequantized_conv2d_bias.data()));
  } else {
    NNADAPTER_CHECK_EQ(conv2d_bias_operand->type.precision, NNADAPTER_FLOAT32);
    memcpy(dequantized_conv2d_bias.data(),
           reinterpret_cast<float*>(conv2d_bias_operand->buffer),
           conv2d_bias_operand->length);
  }
  std::vector<float> dequantized_add_input(conv2d_output_channel_size);
  if (IsInt8SymmPerLayerQuantType(add_input_operand->type.precision)) {
    NNADAPTER_CHECK(DequantizeData<int8_t>(
        reinterpret_cast<int8_t*>(add_input_operand->buffer),
        &conv2d_output_channel_size,
        1,
        &add_input_operand->type.symm_per_layer_params.scale,
        NULL,
        -1,
        -128,
        127,
        dequantized_add_input.data()));
  } else if (IsInt32SymmPerLayerQuantType(add_input_operand->type.precision)) {
    NNADAPTER_CHECK(DequantizeData<int32_t>(
        reinterpret_cast<int32_t*>(add_input_operand->buffer),
        &conv2d_output_channel_size,
        1,
        &add_input_operand->type.symm_per_layer_params.scale,
        NULL,
        -1,
        -128,
        127,
        dequantized_add_input.data()));
  } else {
    NNADAPTER_CHECK_EQ(add_input_operand->type.precision, NNADAPTER_FLOAT32);
    memcpy(dequantized_add_input.data(),
           reinterpret_cast<float*>(add_input_operand->buffer),
           add_input_operand->length);
  }
  // Add the constant input operand of NNADAPTER_ADD
  for (int64_t i = 0; i < conv2d_output_channel_size; i++) {
    dequantized_conv2d_bias[i] += dequantized_add_input[i];
  }
  // Requantize the bias operand of NNADAPTER_CONV_2D
  if (IsInt32SymmPerLayerQuantType(conv2d_bias_operand->type.precision)) {
    NNADAPTER_CHECK(QuantizeData<int32_t>(
        dequantized_conv2d_bias.data(),
        &conv2d_output_channel_size,
        1,
        &conv2d_bias_operand->type.symm_per_layer_params.scale,
        NULL,
        -1,
        -2147483647,
        2147483647,
        reinterpret_cast<int32_t*>(conv2d_bias_operand->buffer)));
  } else if (IsInt32SymmPerChannelQuantType(
                 conv2d_bias_operand->type.precision)) {
    NNADAPTER_CHECK(QuantizeData<int32_t>(
        dequantized_conv2d_bias.data(),
        &conv2d_output_channel_size,
        1,
        conv2d_bias_operand->type.symm_per_channel_params.scales,
        NULL,
        conv2d_bias_operand->type.symm_per_channel_params.channel_dim,
        -2147483647,
        2147483647,
        reinterpret_cast<int32_t*>(conv2d_bias_operand->buffer)));
  } else {
    NNADAPTER_CHECK_EQ(conv2d_bias_operand->type.precision, NNADAPTER_FLOAT32);
    memcpy(reinterpret_cast<float*>(conv2d_bias_operand->buffer),
           dequantized_conv2d_bias.data(),
           conv2d_bias_operand->length);
  }
  // Update the value of the fuse code operand of NNADAPTER_CONV_2D with the
  // value of the fuse code operand of NNADAPTER_ADD.
  *reinterpret_cast<int32_t*>(conv2d_fuse_code_operand->buffer) =
      *reinterpret_cast<int32_t*>(add_fuse_code_operand->buffer);
  // Replace the output operand of NNADAPTER_CONV_2D with the output operand of
  // NNADAPTER_ADD.
  conv2d_operation->output_operands[0] = add_output_operand;
  // The matched intermediate operands and operations will be deleted only when
  // it returns true.
  return true;
}

NNADAPTER_EXPORT void FuseConv2DAddIntoConv2D(core::Model* model) {
  for (auto conv2d_type : {NNADAPTER_CONV_2D, NNADAPTER_CONV_2D_TRANSPOSE}) {
    NNADAPTER_VLOG(5) << "Apply Conv2DAddFuser for conv2d_type:"
                      << OperationTypeToString(conv2d_type);
    bool stop;
    do {
      Conv2DAddFuser conv2d_add_fuser(conv2d_type);
      stop = conv2d_add_fuser.Apply(model) == 0;
    } while (!stop);
  }
}

}  // namespace nnadapter
