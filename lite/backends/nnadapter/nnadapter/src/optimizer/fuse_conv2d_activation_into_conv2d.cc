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

#include "optimizer/fuse_conv2d_activation_into_conv2d.h"
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

class Conv2DActivationFuser : public PatternMatcher {
 public:
  explicit Conv2DActivationFuser(NNAdapterOperationType conv2d_type,
                                 NNAdapterOperationType activation_type)
      : conv2d_type_(conv2d_type), activation_type_(activation_type) {}
  void BuildPattern() override;
  bool HandleMatchedResults(core::Model* model,
                            const std::map<std::string, Node*>& nodes) override;

 private:
  NNAdapterOperationType conv2d_type_{NNADAPTER_CONV_2D};
  NNAdapterOperationType activation_type_{NNADAPTER_RELU};
};

void Conv2DActivationFuser::BuildPattern() {
  // Operation patterns
  auto conv2d_pattern = CreatePattern("conv2d", conv2d_type_);
  auto activation_pattern =
      CreatePattern("activation", activation_type_)->IsIntermediate();
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
                                   ->IsIntermediate();
  auto activation_output_pattern =
      CreatePattern("activation_output")
          ->IsOperationOutputOperand(activation_type_, 0);
  // Create the topological connections for the above patterns
  std::vector<Pattern*> conv2d_input_patterns{conv2d_input_pattern,
                                              conv2d_fuse_code_pattern};
  conv2d_input_patterns >> *conv2d_pattern >> *conv2d_output_pattern;
  *conv2d_output_pattern >> *activation_pattern >> *activation_output_pattern;
}

bool Conv2DActivationFuser::HandleMatchedResults(
    core::Model* model, const std::map<std::string, Node*>& nodes) {
  // Get the operands and operations from the matched subgraph nodes.
  auto conv2d_operation = nodes.at("conv2d")->operation;
  auto conv2d_fuse_code_operand = nodes.at("conv2d_fuse_code")->operand;
  auto activation_output_operand = nodes.at("activation_output")->operand;
  // Replace the output operand the of NNADAPTER_CONV_2D with the output operand
  // of activation operations
  *reinterpret_cast<int32_t*>(conv2d_fuse_code_operand->buffer) =
      OperationTypeToFuseCode(activation_type_);
  conv2d_operation->output_operands[0] = activation_output_operand;
  // The matched intermediate operands and operations will be deleted only when
  // it returns true.
  return true;
}

NNADAPTER_EXPORT void FuseConv2DActivationIntoConv2D(core::Model* model) {
  for (auto conv2d_type : {NNADAPTER_CONV_2D, NNADAPTER_CONV_2D_TRANSPOSE}) {
    for (auto activation_type : {NNADAPTER_RELU, NNADAPTER_RELU6}) {
      NNADAPTER_VLOG(5) << "Apply Conv2DActivationFuser for conv2d_type:"
                        << OperationTypeToString(conv2d_type)
                        << " activation_type:"
                        << OperationTypeToString(activation_type);
      bool stop;
      do {
        Conv2DActivationFuser conv2d_activation_fuser(conv2d_type,
                                                      activation_type);
        stop = conv2d_activation_fuser.Apply(model) == 0;
      } while (!stop);
    }
  }
}

}  // namespace nnadapter
