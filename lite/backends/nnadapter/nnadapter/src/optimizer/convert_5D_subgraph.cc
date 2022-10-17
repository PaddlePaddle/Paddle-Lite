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

#include <algorithm>
#include <map>
#include <vector>
#include "optimizer/pattern_matcher.h"
#include "optimizer/remove_useless_operations.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {

class ReshapeTransposeSlice3Fuser : public PatternMatcher {
 public:
  ReshapeTransposeSlice3Fuser() {}
  void BuildPattern() override;
  bool HandleMatchedResults(core::Model* model,
                            const std::map<std::string, Node*>& nodes) override;
};

void ReshapeTransposeSlice3Fuser::BuildPattern() {
  // Reshape
  auto reshape_in = CreatePattern("reshape_in")
                        ->IsOperationInputOperand(NNADAPTER_RESHAPE, 0);
  auto reshape_shape =
      CreatePattern("reshape_shape")
          ->IsOperationInputOperand(NNADAPTER_RESHAPE, 1)
          ->MatchCondition([](const Node* node) -> bool {
            return node->operand->length / GetOperandPrecisionDataLength(
                                               node->operand->type.precision) ==
                   5;
          });
  auto reshape = CreatePattern("reshape", NNADAPTER_RESHAPE);
  auto reshape_out = CreatePattern("reshape_out")
                         ->IsOperationOutputOperand(NNADAPTER_RESHAPE, 0)
                         ->IsOperationInputOperand(NNADAPTER_TRANSPOSE, 0)
                         ->CheckOutputCount(1);
  // Transpose
  auto transpose_perm =
      CreatePattern("transpose_perm")
          ->IsOperationInputOperand(NNADAPTER_TRANSPOSE, 1)
          ->MatchCondition([](const Node* node) -> bool {
            auto perm_data = reinterpret_cast<int32_t*>(node->operand->buffer);
            auto perm_count = node->operand->length /
                              GetOperandPrecisionDataLength(NNADAPTER_INT32);
            std::vector<int32_t> perm(perm_data, perm_data + perm_count);
            return perm == std::vector<int32_t>{2, 0, 3, 1, 4};
          });
  auto transpose = CreatePattern("transpose", NNADAPTER_TRANSPOSE);
  auto transpose_out = CreatePattern("transpose_out")
                           ->IsOperationOutputOperand(NNADAPTER_TRANSPOSE, 0)
                           ->IsOperationInputOperand(NNADAPTER_SLICE, 0)
                           ->CheckOutputCount(3);
  // Slice 0,1,2
  auto slice_axes =
      CreatePattern("slice_axes")
          ->IsOperationInputOperand(NNADAPTER_SLICE, 1)
          ->MatchCondition([](const Node* node) -> bool {
            auto perm_data = reinterpret_cast<int32_t*>(node->operand->buffer);
            auto perm_count = node->operand->length /
                              GetOperandPrecisionDataLength(NNADAPTER_INT32);
            std::vector<int32_t> perm(perm_data, perm_data + perm_count);
            return perm == std::vector<int32_t>{2, 0, 3, 1, 4};
          });

  // Slice0

  // Slice1

  // Slice2
}

bool ReshapeTransposeSlice3Fuser::HandleMatchedResults(
    core::Model* model, const std::map<std::string, Node*>& nodes) {
  return true;
}

// Reduce 5D subgraph to satisfies the constraints
// (some op only support 4D at most).
/***************************************************
1. Origin subgraph:
                reshape (3D->5D)
                        |
                transpose (5D->5D)
                        |
    |-------------------|-------------------|
slice(5D->5D)      slice(5D->5D)      slice(5D->5D)
    |                   |                   |
squeeze(5D->4D)    squeeze(5D->4D)    squeeze(5D->4D)

***************************************************
2. Optimized subgraph:
                reshape (3D->4D)
                        |
                transpose (4D->4D)
                        |
    |-------------------|-------------------|
slice(4D->4D)      slice(4D->4D)      slice(4D->4D)

***************************************************/
void FuseReshapeTransposeSlice3(core::Model* model) {}

}  // namespace nnadapter
