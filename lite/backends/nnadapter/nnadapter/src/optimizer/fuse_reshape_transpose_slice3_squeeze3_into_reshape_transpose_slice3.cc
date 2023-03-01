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

#include "optimizer/fuse_reshape_transpose_slice3_squeeze3_into_reshape_transpose_slice3.h"
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

class ReshapeTransposeSlice3Squeeze3Fuser : public PatternMatcher {
 public:
  ReshapeTransposeSlice3Squeeze3Fuser() {}
  void BuildPattern() override;
  bool HandleMatchedResults(core::Model* model,
                            const std::map<std::string, Node*>& nodes) override;
};

void ReshapeTransposeSlice3Squeeze3Fuser::BuildPattern() {
  // Reshape
  auto reshape_in = CreatePattern("reshape_in")
                        ->IsOperationInputOperand(NNADAPTER_RESHAPE, 0)
                        ->MatchCondition([](const Node* node) -> bool {
                          return !IsOperandWithDynamicShape(node->operand);
                        });
  auto reshape_shape =
      CreatePattern("reshape_shape")
          ->IsOperationInputOperand(NNADAPTER_RESHAPE, 1)
          ->MatchCondition([](const Node* node) -> bool {
            auto precision_data_length =
                GetOperandPrecisionDataLength(node->operand->type.precision);
            return node->operand->length / precision_data_length == 5;
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
#define ADD_SLICE(id_, start_, end_)                                           \
  auto slice##id_##_axes =                                                     \
      CreatePattern("slice" #id_ "_axes")                                      \
          ->IsOperationInputOperand(NNADAPTER_SLICE, 1)                        \
          ->MatchCondition([](const Node* node) -> bool {                      \
            auto axes_data =                                                   \
                reinterpret_cast<int32_t*>(node->operand->buffer);             \
            auto axes_count = node->operand->length /                          \
                              GetOperandPrecisionDataLength(NNADAPTER_INT32);  \
            return axes_count == 1 && axes_data[0] == 0;                       \
          });                                                                  \
  auto slice##id_##_starts =                                                   \
      CreatePattern("slice" #id_ "_starts")                                    \
          ->IsOperationInputOperand(NNADAPTER_SLICE, 2)                        \
          ->MatchCondition([](const Node* node) -> bool {                      \
            auto starts_data =                                                 \
                reinterpret_cast<int32_t*>(node->operand->buffer);             \
            auto starts_count =                                                \
                node->operand->length /                                        \
                GetOperandPrecisionDataLength(NNADAPTER_INT32);                \
            return starts_count == 1 && starts_data[0] == start_;              \
          });                                                                  \
  auto slice##id_##_ends =                                                     \
      CreatePattern("slice" #id_ "_ends")                                      \
          ->IsOperationInputOperand(NNADAPTER_SLICE, 3)                        \
          ->MatchCondition([](const Node* node) -> bool {                      \
            auto ends_data =                                                   \
                reinterpret_cast<int32_t*>(node->operand->buffer);             \
            auto ends_count = node->operand->length /                          \
                              GetOperandPrecisionDataLength(NNADAPTER_INT32);  \
            return ends_count == 1 && ends_data[0] == end_;                    \
          });                                                                  \
  auto slice##id_##_steps =                                                    \
      CreatePattern("slice" #id_ "_steps")                                     \
          ->IsOperationInputOperand(NNADAPTER_SLICE, 4)                        \
          ->MatchCondition([](const Node* node) -> bool {                      \
            auto steps_data =                                                  \
                reinterpret_cast<int32_t*>(node->operand->buffer);             \
            auto steps_count = node->operand->length /                         \
                               GetOperandPrecisionDataLength(NNADAPTER_INT32); \
            return steps_count == 1 && steps_data[0] == 1;                     \
          });                                                                  \
  auto slice##id_ = CreatePattern("slice" #id_, NNADAPTER_SLICE);              \
  auto slice##id_##_out = CreatePattern("slice" #id_ "_out")                   \
                              ->IsOperationOutputOperand(NNADAPTER_SLICE, 0)   \
                              ->IsOperationInputOperand(NNADAPTER_SQUEEZE, 0)  \
                              ->CheckOutputCount(1)                            \
                              ->IsIntermediate();
  ADD_SLICE(0, 0, 1);
  ADD_SLICE(1, 1, 2);
  ADD_SLICE(2, 2, 3);
#undef ADD_SLICE

// Squeeze 0,1,2
#define ADD_SQUEEZE(id_)                                                      \
  auto squeeze##id_##_axes =                                                  \
      CreatePattern("squeeze" #id_ "_axes")                                   \
          ->IsOperationInputOperand(NNADAPTER_SQUEEZE, 1)                     \
          ->MatchCondition([](const Node* node) -> bool {                     \
            auto axes_data =                                                  \
                reinterpret_cast<int32_t*>(node->operand->buffer);            \
            auto axes_count = node->operand->length /                         \
                              GetOperandPrecisionDataLength(NNADAPTER_INT32); \
            return axes_count == 1 && axes_data[0] == 0;                      \
          })                                                                  \
          ->IsIntermediate();                                                 \
  auto squeeze##id_ =                                                         \
      CreatePattern("squeeze" #id_, NNADAPTER_SQUEEZE)->IsIntermediate();     \
  auto squeeze##id_##_out =                                                   \
      CreatePattern("squeeze" #id_ "_out")                                    \
          ->IsOperationOutputOperand(NNADAPTER_SQUEEZE, 0);
  ADD_SQUEEZE(0);
  ADD_SQUEEZE(1);
  ADD_SQUEEZE(2);
#undef ADD_SQUEEZE

  // Create the topological connections for the above patterns
  std::vector<Pattern*> reshape_ins{reshape_in, reshape_shape};
  reshape_ins >> *reshape >> *reshape_out;
  std::vector<Pattern*> transpose_ins{reshape_out, transpose_perm};
  transpose_ins >> *transpose >> *transpose_out;
#define SLICE_SQUEEZE_CONNECTIONS(id_)                           \
  std::vector<Pattern*> slice##id_##_ins{transpose_out,          \
                                         slice##id_##_axes,      \
                                         slice##id_##_starts,    \
                                         slice##id_##_ends,      \
                                         slice##id_##_steps};    \
  slice##id_##_ins >> *slice##id_ >> *slice##id_##_out;          \
  std::vector<Pattern*> squeeze##id_##_ins{slice##id_##_out,     \
                                           squeeze##id_##_axes}; \
  squeeze##id_##_ins >> *squeeze##id_ >> *squeeze##id_##_out;
  SLICE_SQUEEZE_CONNECTIONS(0);
  SLICE_SQUEEZE_CONNECTIONS(1);
  SLICE_SQUEEZE_CONNECTIONS(2);
#undef SLICE_SQUEEZE_CONNECTIONS
}

bool ReshapeTransposeSlice3Squeeze3Fuser::HandleMatchedResults(
    core::Model* model, const std::map<std::string, Node*>& nodes) {
  // Modify reshape
  auto reshape_out_operand = nodes.at("reshape_out")->operand;
  auto reshape_out_shape_data = reshape_out_operand->type.dimensions.data;
  reshape_out_shape_data[2] *= reshape_out_shape_data[3];
  reshape_out_shape_data[3] = reshape_out_shape_data[4];
  auto reshape_out_shape_count = 4;
  reshape_out_operand->type.dimensions.count = reshape_out_shape_count;
  auto reshape_shape_operand = nodes.at("reshape_shape")->operand;
  auto reshape_shape_data =
      reinterpret_cast<int32_t*>(reshape_shape_operand->buffer);
  memcpy(reshape_shape_data,
         reshape_out_shape_data,
         reshape_out_shape_count * sizeof(int32_t));
  reshape_shape_operand->type.precision = NNADAPTER_INT32;
  reshape_shape_operand->length = reshape_out_shape_count * sizeof(int32_t);

  // Modify transpose
  auto transpose_perm = nodes.at("transpose_perm")->operand;
  auto transpose_perm_data = reinterpret_cast<int32_t*>(transpose_perm->buffer);
  transpose_perm_data[0] = 0;
  transpose_perm_data[1] = 2;
  transpose_perm_data[2] = 1;
  transpose_perm_data[3] = 3;
  transpose_perm->length = 4 * sizeof(int32_t);
  auto transpose_out_operand = nodes.at("transpose_out")->operand;
  transpose_out_operand->type.dimensions.count = reshape_out_shape_count;
  auto transpose_out_shape_data = transpose_out_operand->type.dimensions.data;
  transpose_out_shape_data[0] = reshape_out_shape_data[0];
  transpose_out_shape_data[1] = reshape_out_shape_data[2];
  transpose_out_shape_data[2] = reshape_out_shape_data[1];
  transpose_out_shape_data[3] = reshape_out_shape_data[3];

  // Modify slice 0,1,2
  int slice_length = transpose_out_shape_data[1] / 3;
#define MODIFY_SLICE(id_)                                      \
  reinterpret_cast<int32_t*>(                                  \
      nodes.at("slice" #id_ "_axes")->operand->buffer)[0] = 1; \
  reinterpret_cast<int32_t*>(                                  \
      nodes.at("slice" #id_ "_starts")->operand->buffer)[0] =  \
      id_ * slice_length;                                      \
  reinterpret_cast<int32_t*>(                                  \
      nodes.at("slice" #id_ "_ends")->operand->buffer)[0] =    \
      (id_ + 1) * slice_length;                                \
  nodes.at("slice" #id_)->operation->output_operands[0] =      \
      nodes.at("squeeze" #id_ "_out")->operand;
  MODIFY_SLICE(0);
  MODIFY_SLICE(1);
  MODIFY_SLICE(2);
#undef MODIFY_SLICE

  return true;
}

/***************************************************
A repeating structure in "vit" model
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
void FuseReshapeTransposeSlice3Squeeze3IntoReshapeTransposeSlice3(
    core::Model* model) {
  NNADAPTER_VLOG(5) << "Apply ReshapeTransposeSlice3Squeeze3Fuser";
  bool stop;
  do {
    ReshapeTransposeSlice3Squeeze3Fuser fuser;
    stop = fuser.Apply(model) == 0;
  } while (!stop);
}

}  // namespace nnadapter
