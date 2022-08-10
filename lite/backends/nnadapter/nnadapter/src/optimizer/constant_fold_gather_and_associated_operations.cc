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

#include "optimizer/constant_fold_gather_and_associated_operations.h"
#include "optimizer/pattern_matcher.h"
#include "utility/debug.h"
#include "utility/hints.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {

/*
before:
  -> fill_like -> mul -> gather -> add ->
-------------------------------------------------------
after:
  -> add ->
*/
class FillLikeMulGatherAddFuser : public PatternMatcher {
 public:
  FillLikeMulGatherAddFuser() {}
  void BuildPattern() override;
  bool HandleMatchedResults(core::Model* model,
                            const std::map<std::string, Node*>& nodes) override;
};

void FillLikeMulGatherAddFuser::BuildPattern() {
  // Create patterns
  auto fill_like_in = CreatePattern("fill_like_in")
                          ->IsOperationInputOperand(NNADAPTER_FILL_LIKE, 0)
                          ->MatchCondition([](const Node* node) -> bool {
                            auto operand = node->operand;
                            return operand != nullptr &&
                                   !IsDynamicShapeOperandType(operand->type);
                          });
  auto fill_like_value = CreatePattern("fill_like_value")
                             ->IsOperationInputOperand(NNADAPTER_FILL_LIKE, 1)
                             ->IsIntermediate();
  auto fill_like =
      CreatePattern("fill_like", NNADAPTER_FILL_LIKE)->IsIntermediate();
  auto fill_like_out = CreatePattern("fill_like_out")
                           ->IsOperationOutputOperand(NNADAPTER_FILL_LIKE, 0)
                           ->IsOperationInputOperand(NNADAPTER_MUL, 0)
                           ->IsIntermediate();
  auto mul_y = CreatePattern("mul_y")
                   ->IsConstantOperand()
                   ->IsOperationInputOperand(NNADAPTER_MUL, 1)
                   ->MatchCondition([](const Node* node) -> bool {
                     auto operand = node->operand;
                     return operand != nullptr &&
                            operand->type.precision == NNADAPTER_INT32 &&
                            operand->length == sizeof(int32_t) &&
                            *reinterpret_cast<int32_t*>(operand->buffer) == 0;
                   })
                   ->IsIntermediate();
  auto mul_fuse_code = CreatePattern("mul_fuse_code")
                           ->IsOperationInputOperand(NNADAPTER_MUL, 2)
                           ->IsIntermediate();
  auto mul = CreatePattern("mul", NNADAPTER_MUL)->IsIntermediate();
  auto mul_out = CreatePattern("mul_out")
                     ->IsOperationOutputOperand(NNADAPTER_MUL, 0)
                     ->IsOperationInputOperand(NNADAPTER_GATHER, 1)
                     ->IsIntermediate();
  auto gather_in = CreatePattern("gather_in")
                       ->IsOperationInputOperand(NNADAPTER_GATHER, 0)
                       ->MatchCondition([](const Node* node) -> bool {
                         auto operand = node->operand;
                         return operand != nullptr &&
                                operand->type.precision == NNADAPTER_FLOAT32;
                       })
                       ->IsIntermediate();
  auto gather_axis = CreatePattern("gather_axis")
                         ->IsOperationInputOperand(NNADAPTER_GATHER, 2)
                         ->IsIntermediate();
  auto gather = CreatePattern("gather", NNADAPTER_GATHER)->IsIntermediate();
  auto gather_out = CreatePattern("gather_out")
                        ->IsOperationOutputOperand(NNADAPTER_GATHER, 0)
                        ->IsOperationInputOperand(NNADAPTER_ADD, 1)
                        ->IsIntermediate();
  auto add_x =
      CreatePattern("add_x")->IsOperationInputOperand(NNADAPTER_ADD, 0);
  auto add = CreatePattern("add", NNADAPTER_ADD);
  // Create the topological connections for the above patterns
  std::vector<Pattern*> fill_like_ins{fill_like_in, fill_like_value};
  fill_like_ins >> *fill_like >> *fill_like_out;
  std::vector<Pattern*> mul_ins{fill_like_out, mul_y, mul_fuse_code};
  mul_ins >> *mul >> *mul_out;
  std::vector<Pattern*> gather_ins{gather_in, mul_out, gather_axis};
  gather_ins >> *gather >> *gather_out;
  std::vector<Pattern*> add_ins{gather_out, add_x};
  add_ins >> *add;
}

bool FillLikeMulGatherAddFuser::HandleMatchedResults(
    core::Model* model, const std::map<std::string, Node*>& nodes) {
  // Calculate constant gather out operand
  auto gather_in = nodes.at("gather_in")->operand;
  int gather_in_dims_count = gather_in->type.dimensions.count;
  int* gather_in_dims_data = gather_in->type.dimensions.data;
  auto gather_axis = *reinterpret_cast<int32_t*>(
      nodes.at("gather")->operation->input_operands[2]->buffer);
  if (gather_axis < 0) {
    gather_axis += static_cast<int32_t>(gather_in->type.dimensions.count);
  }
  int copy_length = 1;
  for (int i = gather_axis + 1; i < gather_in_dims_count; i++) {
    copy_length *= gather_in_dims_data[i];
  }

  auto gather_out = nodes.at("gather_out")->operand;
  int gather_out_dims_count = gather_out->type.dimensions.count;
  int* gather_out_dims_data = gather_out->type.dimensions.data;
  std::vector<int32_t> dims(gather_out_dims_data,
                            gather_out_dims_data + gather_out_dims_count);
  int length = 1;
  for (auto i : dims) {
    length *= i;
  }

  std::vector<float> values(length, 0.f);
  void* src_data = gather_in->buffer;
  float* values_data = values.data();
  for (int i = 0; i < length; i += copy_length) {
    memcpy(values_data, src_data, copy_length * sizeof(float));
    values_data += copy_length;
  }
  auto constant_gather_out =
      AddFloat32ConstantOperand(model, values.data(), dims);
  // Connect constant_gather_out to add
  auto add = nodes.at("add")->operation;
  add->input_operands[1] = constant_gather_out;
  return true;
}

/*
before:
  -> fill_like -> cum_sum -> sub -> gather -> add ->
         \----------/
-------------------------------------------------------
after:
  -> add ->
*/
class FillLikeCumSumSubGatherAddFuser : public PatternMatcher {
 public:
  FillLikeCumSumSubGatherAddFuser() {}
  void BuildPattern() override;
  bool HandleMatchedResults(core::Model* model,
                            const std::map<std::string, Node*>& nodes) override;
};

void FillLikeCumSumSubGatherAddFuser::BuildPattern() {
  // Create patterns
  auto fill_like_in = CreatePattern("fill_like_in")
                          ->IsOperationInputOperand(NNADAPTER_FILL_LIKE, 0)
                          ->MatchCondition([](const Node* node) -> bool {
                            auto operand = node->operand;
                            return operand != nullptr &&
                                   !IsDynamicShapeOperandType(operand->type);
                          });
  auto fill_like =
      CreatePattern("fill_like", NNADAPTER_FILL_LIKE)->IsIntermediate();
  auto fill_like_value =
      CreatePattern("fill_like_value")
          ->IsOperationInputOperand(NNADAPTER_FILL_LIKE, 1)
          ->MatchCondition([](const Node* node) -> bool {
            auto operand = node->operand;
            return operand != nullptr &&
                   operand->type.precision == NNADAPTER_INT32 &&
                   operand->length == sizeof(int32_t) &&
                   *reinterpret_cast<int32_t*>(operand->buffer) == 1;
          })
          ->IsIntermediate();
  auto fill_like_out = CreatePattern("fill_like_out")
                           ->IsOperationOutputOperand(NNADAPTER_FILL_LIKE, 0)
                           ->IsOperationInputOperand(NNADAPTER_CUM_SUM, 0)
                           ->IsOperationInputOperand(NNADAPTER_SUB, 1)
                           ->IsIntermediate();
  auto cum_sum_axis =
      CreatePattern("cum_sum_axis")
          ->IsOperationInputOperand(NNADAPTER_CUM_SUM, 1)
          ->MatchCondition([](const Node* node) -> bool {
            int32_t axis = *reinterpret_cast<int32_t*>(node->operand->buffer);
            return axis == 1 || axis == -1;
          })
          ->IsIntermediate();
  auto cum_sum_exclusive =
      CreatePattern("cum_sum_exclusive")
          ->IsOperationInputOperand(NNADAPTER_CUM_SUM, 2)
          ->MatchCondition([](const Node* node) -> bool {
            auto operand = node->operand;
            return operand != nullptr &&
                   operand->type.precision == NNADAPTER_BOOL8 &&
                   operand->length == sizeof(bool) &&
                   *reinterpret_cast<bool*>(operand->buffer) == false;
          })
          ->IsIntermediate();
  auto cum_sum_reverse =
      CreatePattern("cum_sum_reverse")
          ->IsOperationInputOperand(NNADAPTER_CUM_SUM, 3)
          ->MatchCondition([](const Node* node) -> bool {
            auto operand = node->operand;
            return operand != nullptr &&
                   operand->type.precision == NNADAPTER_BOOL8 &&
                   operand->length == sizeof(bool) &&
                   *reinterpret_cast<bool*>(operand->buffer) == false;
          })
          ->IsIntermediate();
  auto cum_sum = CreatePattern("cum_sum", NNADAPTER_CUM_SUM)->IsIntermediate();
  auto cum_sum_out = CreatePattern("cum_sum_out")
                         ->IsOperationOutputOperand(NNADAPTER_CUM_SUM, 0)
                         ->IsOperationInputOperand(NNADAPTER_SUB, 0)
                         ->IsIntermediate();
  auto sub_fuse_code = CreatePattern("sub_fuse_code")
                           ->IsOperationInputOperand(NNADAPTER_SUB, 2)
                           ->IsIntermediate();
  auto sub = CreatePattern("sub", NNADAPTER_SUB)->IsIntermediate();
  auto sub_out = CreatePattern("sub_out")
                     ->IsOperationOutputOperand(NNADAPTER_SUB, 0)
                     ->IsOperationInputOperand(NNADAPTER_GATHER, 1)
                     ->IsIntermediate();
  auto gather_in = CreatePattern("gather_in")
                       ->IsOperationInputOperand(NNADAPTER_GATHER, 0)
                       ->IsIntermediate();
  auto gather_axis = CreatePattern("gather_axis")
                         ->IsOperationInputOperand(NNADAPTER_GATHER, 2)
                         ->IsIntermediate();
  auto gather = CreatePattern("gather", NNADAPTER_GATHER)->IsIntermediate();
  auto gather_out = CreatePattern("gather_out")
                        ->IsOperationOutputOperand(NNADAPTER_GATHER, 0)
                        ->IsOperationInputOperand(NNADAPTER_ADD, 1)
                        ->IsIntermediate();
  auto add_x =
      CreatePattern("add_x")->IsOperationInputOperand(NNADAPTER_ADD, 0);
  auto add = CreatePattern("add", NNADAPTER_ADD);
  // Create the topological connections for the above patterns
  std::vector<Pattern*> fill_like_ins{fill_like_in, fill_like_value};
  fill_like_ins >> *fill_like >> *fill_like_out;
  std::vector<Pattern*> cum_sum_ins{
      fill_like_out, cum_sum_axis, cum_sum_exclusive, cum_sum_reverse};
  cum_sum_ins >> *cum_sum >> *cum_sum_out;
  std::vector<Pattern*> sub_ins{cum_sum_out, cum_sum_out, sub_fuse_code};
  sub_ins >> *sub >> *sub_out;
  std::vector<Pattern*> gather_ins{gather_in, sub_out, gather_axis};
  gather_ins >> *gather >> *gather_out;
  std::vector<Pattern*> add_ins{gather_out, add_x};
  add_ins >> *add;
}

bool FillLikeCumSumSubGatherAddFuser::HandleMatchedResults(
    core::Model* model, const std::map<std::string, Node*>& nodes) {
  // Calculate constant gather out operand
  auto gather_out = nodes.at("gather_out")->operand;
  int gather_out_dims_count = gather_out->type.dimensions.count;
  int* gather_out_dims_data = gather_out->type.dimensions.data;
  std::vector<int32_t> dims(gather_out_dims_data,
                            gather_out_dims_data + gather_out_dims_count);
  int length = 1;
  for (auto i : dims) {
    length *= i;
  }
  std::vector<float> values(length, 0.f);
  auto gather_in = nodes.at("gather_in")->operand;
  memcpy(values.data(), gather_in->buffer, length * sizeof(float));
  auto constant_gather_out =
      AddFloat32ConstantOperand(model, values.data(), dims);
  // Connect constant_gather_out to add
  auto add = nodes.at("add")->operation;
  add->input_operands[1] = constant_gather_out;
  return true;
}

NNADAPTER_EXPORT void ConstantFoldGatherAndAssociatedOperations(
    core::Model* model) {
  bool stop;

  NNADAPTER_VLOG(5) << "Apply FillLikeMulGatherAddFuser";
  do {
    FillLikeMulGatherAddFuser fuser;
    stop = fuser.Apply(model) == 0;
  } while (!stop);

  NNADAPTER_VLOG(5) << "Apply FillLikeCumSumSubGatherAddFuser";
  do {
    FillLikeCumSumSubGatherAddFuser fuser;
    stop = fuser.Apply(model) == 0;
  } while (!stop);
}

}  // namespace nnadapter
