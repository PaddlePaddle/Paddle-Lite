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

#include "optimizer/constant_fold_shape_and_associated_operations.h"
#include <set>
#include "optimizer/pattern_matcher.h"
#include "utility/debug.h"
#include "utility/hints.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"

namespace nnadapter {

NNADAPTER_EXPORT void ConstantFoldShapeAndAssociatedOperations(
    core::Model *model) {
  NNADAPTER_LOG(INFO) << "Run ConstantFoldShapeAndAssociatedOperations Pass";
  std::vector<core::Operation *> operations =
      SortOperationsInTopologicalOrder(model);
  // Check whether to support dynamic shape in the model
  for (auto operation : operations) {
    auto input_operands = operation->input_operands;
    for (auto operand : input_operands) {
      if (operand && IsModelInputOperand(operand)) {
        break;
      }
      if (operand && IsTemporaryVariableOperand(operand) &&
          IsOperandWithDynamicShape(operand)) {
        NNADAPTER_LOG(WARNING)
            << "Skip if dynamic shape need to be supported in the model!";
        return;
      }
    }
  }
  // Operands that will not be removed
  std::set<core::Operand *> white_operands;
  // Operands and operations to be deleted
  std::set<core::Operand *> remove_operands;
  std::set<core::Operation *> remove_operations;
  // Collect operands and operations that need to be deleted
  for (auto operation : operations) {
    auto input_operands = operation->input_operands;
    auto output_operands = operation->output_operands;
    if (operation->type == NNADAPTER_SHAPE) {
      for (auto operand : output_operands) {
        remove_operands.insert(operand);
      }
      remove_operations.insert(operation);
    } else if (operation->type == NNADAPTER_RESIZE_LINEAR ||
               operation->type == NNADAPTER_RESIZE_NEAREST) {
      white_operands.insert(input_operands[1]);
      white_operands.insert(input_operands[2]);
    } else if (operation->type == NNADAPTER_EXPAND ||
               operation->type == NNADAPTER_RESHAPE) {
      white_operands.insert(input_operands[1]);
    } else {
      bool is_tempory_shape_op = true;
      for (auto input_operand : input_operands) {
        if (IsTemporaryVariableOperand(input_operand) ||
            IsModelInputOperand(input_operand)) {
          is_tempory_shape_op = false;
          break;
        }
      }
      if (!is_tempory_shape_op) continue;
      for (auto output_operand : output_operands) {
        if (IsTemporaryVariableOperand(output_operand) ||
            IsModelOutputOperand(output_operand)) {
          is_tempory_shape_op = false;
          break;
        }
      }
      if (is_tempory_shape_op) {
        for (auto operand : input_operands) {
          remove_operands.insert(operand);
        }
        for (auto operand : output_operands) {
          remove_operands.insert(operand);
        }
        remove_operations.insert(operation);
      }
    }
  }
  // The operations cannot be deleted completely
  if (operations.size() == remove_operations.size()) {
    NNADAPTER_LOG(WARNING)
        << "Skip! The operations cannot be deleted completely.";
    return;
  }
  // Clean
  for (auto remove_operand : remove_operands) {
    if (!white_operands.count(remove_operand)) {
      RemoveOperand(model, remove_operand);
      NNADAPTER_VLOG(5) << "Operand: " << OperandIdToString(remove_operand)
                        << " is deleted!";
    } else {
      auto &temporary_shape = *(GetTemporaryShape(remove_operand));
      auto precision = remove_operand->type.precision;
      if (precision == NNADAPTER_INT32) {
        remove_operand->length =
            temporary_shape.count * static_cast<uint32_t>(sizeof(int32_t));
      } else if (precision == NNADAPTER_FLOAT32) {
        remove_operand->length =
            temporary_shape.count * static_cast<uint32_t>(sizeof(float));
      } else if (precision == NNADAPTER_INT64) {
        remove_operand->length =
            temporary_shape.count * static_cast<uint32_t>(sizeof(int64_t));
      }
      remove_operand->type.lifetime = NNADAPTER_CONSTANT_COPY;
      remove_operand->buffer = malloc(remove_operand->length);
      memcpy(
          remove_operand->buffer, temporary_shape.data, remove_operand->length);
      NNADAPTER_VLOG(5) << "Operand: " << OperandIdToString(remove_operand)
                        << " is in constant folding!";
    }
  }
  for (auto remove_operation : remove_operations) {
    RemoveOperation(model, remove_operation);
    NNADAPTER_VLOG(5) << "Operation: "
                      << OperationTypeToString(remove_operation->type)
                      << " is deleted!";
  }
}

/*
before:
  shape    fill_like    fill_like
    |          |            |
  slice        |            |
    \----------|------------/
            concat
               |
             tile

after:
   tile
*/
class FoldShapeSliceConcatTileFuser : public PatternMatcher {
 public:
  FoldShapeSliceConcatTileFuser() {}
  void BuildPattern() override;
  bool HandleMatchedResults(
      core::Model *model, const std::map<std::string, Node *> &nodes) override;
};

void FoldShapeSliceConcatTileFuser::BuildPattern() {
  // Create patterns
  auto shape_in =
      CreatePattern("shape_in")->IsOperationInputOperand(NNADAPTER_SHAPE, 0);
  auto shape_dtype = CreatePattern("shape_dtype")
                         ->IsOperationInputOperand(NNADAPTER_SHAPE, 1)
                         ->IsIntermediate();
  auto shape = CreatePattern("shape", NNADAPTER_SHAPE)->IsIntermediate();
  auto shape_out = CreatePattern("shape_out")
                       ->IsOperationOutputOperand(NNADAPTER_SHAPE, 0)
                       ->IsOperationInputOperand(NNADAPTER_SLICE, 0)
                       ->IsIntermediate();

  auto slice_axes = CreatePattern("slice_axes")
                        ->IsOperationInputOperand(NNADAPTER_SLICE, 1)
                        ->IsIntermediate();
  auto slice_starts = CreatePattern("slice_starts")
                          ->IsOperationInputOperand(NNADAPTER_SLICE, 2)
                          ->IsIntermediate();
  auto slice_ends = CreatePattern("slice_ends")
                        ->IsOperationInputOperand(NNADAPTER_SLICE, 3)
                        ->IsIntermediate();
  auto slice_steps = CreatePattern("slice_steps")
                         ->IsOperationInputOperand(NNADAPTER_SLICE, 4)
                         ->IsIntermediate();
  auto slice = CreatePattern("slice", NNADAPTER_SLICE)->IsIntermediate();
  auto slice_out = CreatePattern("slice_out")
                       ->IsOperationOutputOperand(NNADAPTER_SLICE, 0)
                       ->IsOperationInputOperand(NNADAPTER_CONCAT, 1)
                       ->IsIntermediate();

  auto concat_in0 = CreatePattern("concat_in0")
                        ->IsConstantOperand()
                        ->IsOperationInputOperand(NNADAPTER_CONCAT, 0)
                        ->IsIntermediate();
  auto concat_in2 = CreatePattern("concat_in2")
                        ->IsConstantOperand()
                        ->IsOperationInputOperand(NNADAPTER_CONCAT, 2)
                        ->IsIntermediate();
  auto concat_axis = CreatePattern("concat_axis")
                         ->IsOperationInputOperand(NNADAPTER_CONCAT, 3)
                         ->IsIntermediate();
  auto concat = CreatePattern("concat", NNADAPTER_CONCAT)->IsIntermediate();
  auto concat_out = CreatePattern("concat_out")
                        ->IsOperationOutputOperand(NNADAPTER_CONCAT, 0)
                        ->IsOperationInputOperand(NNADAPTER_TILE, 1)
                        ->IsIntermediate();

  auto tile = CreatePattern("tile", NNADAPTER_TILE);

  // Create the topological connections for the above patterns
  std::vector<Pattern *> shape_ins{shape_in, shape_dtype};
  shape_ins >> *shape >> *shape_out;
  std::vector<Pattern *> slice_ins{
      shape_out, slice_axes, slice_starts, slice_ends, slice_steps};
  slice_ins >> *slice >> *slice_out;
  std::vector<Pattern *> concat_ins{
      concat_in0, slice_out, concat_in2, concat_axis};
  concat_ins >> *concat >> *concat_out >> *tile;
}

bool FoldShapeSliceConcatTileFuser::HandleMatchedResults(
    core::Model *model, const std::map<std::string, Node *> &nodes) {
  auto shape_in_data = nodes.at("shape_in")->operand->type.dimensions.data;
  auto start_index = *reinterpret_cast<int32_t *>(
      nodes.at("slice")->operation->input_operands[2]->buffer);
  std::vector<int32_t> repeat_times{
      *reinterpret_cast<int32_t *>(nodes.at("concat_in0")->operand->buffer),
      shape_in_data[start_index],
      *reinterpret_cast<int32_t *>(nodes.at("concat_in2")->operand->buffer),
  };
  std::vector<int32_t> dims{3};
  auto tile_in1 = AddInt32ConstantOperand(model, repeat_times.data(), dims);
  nodes.at("tile")->operation->input_operands[1] = tile_in1;
  return true;
}

NNADAPTER_EXPORT void FoldShapeSliceConcatTile(core::Model *model) {
  NNADAPTER_VLOG(5) << "Apply FoldShapeSliceConcatTileFuser";
  bool stop;
  do {
    FoldShapeSliceConcatTileFuser fold_shape_slice_concat_tile_fuser;
    stop = fold_shape_slice_concat_tile_fuser.Apply(model) == 0;
  } while (!stop);
}

}  // namespace nnadapter
