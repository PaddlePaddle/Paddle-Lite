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

#include "optimizer/convert_adaptive_pool2d_into_pool2d.h"
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

class AdaptivePool2dIntoPool2dConverter : public PatternMatcher {
 public:
  AdaptivePool2dIntoPool2dConverter() {}
  void BuildPattern() override;
  bool HandleMatchedResults(core::Model* model,
                            const std::map<std::string, Node*>& nodes) override;
};

void AdaptivePool2dIntoPool2dConverter::BuildPattern() {
  // Operation patterns
  auto adaptive_pool2d_pattern =
      CreatePattern("adaptive_pool2d", NNADAPTER_ADAPTIVE_AVERAGE_POOL_2D)
          ->IsIntermediate();
  // Operand patterns
  auto adaptive_pool2d_input_pattern =
      CreatePattern("adaptive_pool2d_input")
          ->IsOperationInputOperand(NNADAPTER_ADAPTIVE_AVERAGE_POOL_2D, 0);
  auto adaptive_pool2d_kernel_size_pattern =
      CreatePattern("adaptive_pool2d_kernel_size")
          ->IsOperationInputOperand(NNADAPTER_ADAPTIVE_AVERAGE_POOL_2D, 1);
  auto adaptive_pool2d_output_pattern =
      CreatePattern("adaptive_pool2d_output")
          ->IsOperationOutputOperand(NNADAPTER_ADAPTIVE_AVERAGE_POOL_2D, 0);
  // Create the topological ,connections for the above patterns
  std::vector<Pattern*> adaptive_pool2d_input_patterns{
      adaptive_pool2d_input_pattern, adaptive_pool2d_kernel_size_pattern};
  adaptive_pool2d_input_patterns >> *adaptive_pool2d_pattern >>
      *adaptive_pool2d_output_pattern;
}

bool AdaptivePool2dIntoPool2dConverter::HandleMatchedResults(
    core::Model* model, const std::map<std::string, Node*>& nodes) {
  NNADAPTER_VLOG(1) << "ysysysys match!!!! AdaptivePool2dIntoPool2dConverter";
  // Get the operands and operations from the matched subgraph nodes.
  auto adaptive_pool2d_input_operand =
      nodes.at("adaptive_pool2d_input")->operand;
  auto adaptive_pool2d_kernel_size_operand =
      nodes.at("adaptive_pool2d_kernel_size")->operand;
  auto adaptive_pool2d_output_operand =
      nodes.at("adaptive_pool2d_output")->operand;
  auto input_height = adaptive_pool2d_input_operand->type.dimensions.data[2];
  auto input_width = adaptive_pool2d_input_operand->type.dimensions.data[3];
  auto output_height = reinterpret_cast<int32_t*>(
      adaptive_pool2d_kernel_size_operand->buffer)[0];
  auto output_width = reinterpret_cast<int32_t*>(
      adaptive_pool2d_kernel_size_operand->buffer)[1];
  core::Operand* auto_pad_operand = AddInt32ConstantOperand(
      model, static_cast<int32_t>(NNADAPTER_AUTO_PAD_NONE));
  std::vector<int32_t> paddings = {0, 0, 0, 0};
  core::Operand* pads_operand = AddInt32ConstantOperand(model, paddings);
  core::Operand* ceil_mode_operand = AddBool8ConstantOperand(model, false);
  core::Operand* count_include_pad_operand =
      AddBool8ConstantOperand(model, true);
  core::Operand* fuse_code_operand = AddInt32ConstantOperand(
      model, static_cast<int32_t>(NNADAPTER_FUSED_NONE));
  auto stride_height = std::floor(input_height / output_height);
  auto stride_width = std::floor(input_width / output_width);
  std::vector<int32_t> strides = {stride_height, stride_width};
  core::Operand* strides_operand =
      AddInt32ConstantOperand(model, strides.data(), {strides.size()});
  // Calulate the kernel size
  int32_t kernel_height = input_height - ((output_height - 1) * stride_height);
  int32_t kernel_width = input_width - ((output_width - 1) * stride_width);
  std::vector<int32_t> kernel_sizes = {kernel_height, kernel_width};
  core::Operand* kernel_size_operand = AddInt32ConstantOperand(
      model, kernel_sizes.data(), {kernel_sizes.size()});
  auto* pool2d_operation = AddOperation(model);
  pool2d_operation->type = NNADAPTER_AVERAGE_POOL_2D;
  pool2d_operation->input_operands = {adaptive_pool2d_input_operand,
                                      auto_pad_operand,
                                      pads_operand,
                                      kernel_size_operand,
                                      strides_operand,
                                      ceil_mode_operand,
                                      count_include_pad_operand,
                                      fuse_code_operand};
  pool2d_operation->output_operands = {adaptive_pool2d_output_operand};
  // The matched intermediate operands and operations will be deleted only when
  // it returns true.
  return true;
}

NNADAPTER_EXPORT void ConvertAdaptivePool2dIntoPool2d(core::Model* model) {
  NNADAPTER_VLOG(5) << "Apply AdaptivePool2dIntoPool2dConverter";
  bool stop;
  do {
    AdaptivePool2dIntoPool2dConverter adaptive_pool2d_converter;
    stop = adaptive_pool2d_converter.Apply(model) == 0;
  } while (!stop);
}

}  // namespace nnadapter
