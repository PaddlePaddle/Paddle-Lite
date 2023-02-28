// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "optimizer/remove_useless_operations.h"
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

// Remove cast if its input and output have the same data_type.
class RemoveUselessCastFuser : public PatternMatcher {
 public:
  RemoveUselessCastFuser() {}
  void BuildPattern() override;
  bool HandleMatchedResults(core::Model* model,
                            const std::map<std::string, Node*>& nodes) override;
};

void RemoveUselessCastFuser::BuildPattern() {
  // Create patterns
  CreatePattern("cast", NNADAPTER_CAST);
}

bool RemoveUselessCastFuser::HandleMatchedResults(
    core::Model* model, const std::map<std::string, Node*>& nodes) {
  auto cast = nodes.at("cast")->operation;
  auto cast_in = cast->input_operands[0];
  auto cast_out = cast->output_operands[0];
  if (cast_in->type.precision != cast_out->type.precision ||
      (IsModelInputOperand(cast_in) && IsModelOutputOperand(cast_out)))
    return false;

  if (IsModelOutputOperand(cast_out)) {
    auto cast_in_producer = GetOperandProducer(model, cast_in);
    for (auto& operand : cast_in_producer->output_operands) {
      if (operand == cast_in) {
        operand = cast_out;
      }
    }
    RemoveOperand(model, cast_in);
  } else {
    auto cast_out_consumers = GetOperandConsumers(model, cast_out);
    for (auto consumer : cast_out_consumers) {
      for (auto& operand : consumer->input_operands) {
        if (operand == cast_out) {
          operand = cast_in;
        }
      }
    }
    RemoveOperand(model, cast_out);
  }

  RemoveOperand(model, cast->input_operands[1]);
  RemoveOperation(model, cast);
  return true;
}

NNADAPTER_EXPORT void RemoveUselessCast(core::Model* model) {
  NNADAPTER_VLOG(5) << "Apply RemoveUselessCastFuser";
  bool stop;
  do {
    RemoveUselessCastFuser remove_useless_cast_fuser;
    stop = remove_useless_cast_fuser.Apply(model) == 0;
  } while (!stop);
}

// Remove mul if its input1 only has value 1.
class RemoveUselessMulFuser : public PatternMatcher {
 public:
  RemoveUselessMulFuser() {}
  void BuildPattern() override;
  bool HandleMatchedResults(core::Model* model,
                            const std::map<std::string, Node*>& nodes) override;
};

void RemoveUselessMulFuser::BuildPattern() {
  // Create patterns
  auto mul_x =
      CreatePattern("mul_x")->IsOperationInputOperand(NNADAPTER_MUL, 0);
  auto mul_y =
      CreatePattern("mul_y")
          ->IsOperationInputOperand(NNADAPTER_MUL, 1)
          ->MatchCondition([](const Node* node) -> bool {
            auto operand = node->operand;
            if (!IsConstantOperand(operand)) return false;
            auto precision = operand->type.precision;
            switch (precision) {
              case NNADAPTER_FLOAT32: {
                int length = operand->length / sizeof(float);
                float* data = reinterpret_cast<float*>(operand->buffer);
                for (int i = 0; i < length; i++) {
                  if (fabs(data[i] - 1.f) > 1e-5) {
                    return false;
                  }
                }
                return true;
              } break;
              case NNADAPTER_INT32: {
                int length = operand->length / sizeof(int32_t);
                int32_t* data = reinterpret_cast<int32_t*>(operand->buffer);
                for (int i = 0; i < length; i++) {
                  if (data[i] != 1) {
                    return false;
                  }
                }
                return true;
              } break;
              default:
                return false;
                break;
            }
            return false;
          })
          ->IsIntermediate();
  auto mul_fuse_code = CreatePattern("mul_fuse_code")
                           ->IsOperationInputOperand(NNADAPTER_MUL, 2)
                           ->MatchCondition([](const Node* node) -> bool {
                             auto operand = node->operand;
                             int fuse_code =
                                 *reinterpret_cast<int32_t*>(operand->buffer);
                             return fuse_code == NNADAPTER_FUSED_NONE;
                           })
                           ->IsIntermediate();
  auto mul = CreatePattern("mul", NNADAPTER_MUL)->IsIntermediate();
  auto mul_out = CreatePattern("mul_out");
  // Create the topological connections for the above patterns
  std::vector<Pattern*> mul_ins{mul_x, mul_y, mul_fuse_code};
  mul_ins >> *mul >> *mul_out;
}

bool RemoveUselessMulFuser::HandleMatchedResults(
    core::Model* model, const std::map<std::string, Node*>& nodes) {
  auto mul_x = nodes.at("mul_x")->operand;
  auto mul_out = nodes.at("mul_out")->operand;
  if (IsModelInputOperand(mul_x) && IsModelOutputOperand(mul_out)) return false;

  if (IsModelOutputOperand(mul_out)) {
    auto mul_x_consumers = GetOperandConsumers(model, mul_x);
    if (mul_x_consumers.size() > 1) return false;
    auto mul_x_producer = GetOperandProducer(model, mul_x);
    for (auto& operand : mul_x_producer->output_operands) {
      if (operand == mul_x) {
        operand = mul_out;
      }
    }
    RemoveOperand(model, mul_x);
  } else {
    auto mul_out_consumers = GetOperandConsumers(model, mul_out);
    for (auto consumer : mul_out_consumers) {
      for (auto& operand : consumer->input_operands) {
        if (operand == mul_out) {
          operand = mul_x;
        }
      }
    }
    RemoveOperand(model, mul_out);
  }
  return true;
}

NNADAPTER_EXPORT void RemoveUselessMul(core::Model* model) {
  NNADAPTER_VLOG(5) << "Apply RemoveUselessMulFuser";
  bool stop;
  do {
    RemoveUselessMulFuser remove_useless_mul_fuser;
    stop = remove_useless_mul_fuser.Apply(model) == 0;
  } while (!stop);
}

}  // namespace nnadapter
