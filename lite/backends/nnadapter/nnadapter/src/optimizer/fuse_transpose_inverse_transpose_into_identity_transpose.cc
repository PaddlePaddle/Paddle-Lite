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

#include "optimizer/fuse_transpose_inverse_transpose_into_identity_transpose.h"
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

class TransposeInverseTransposeFuser : public PatternMatcher {
 public:
  TransposeInverseTransposeFuser() {}
  void BuildPattern() override;
  bool HandleMatchedResults(core::Model* model,
                            const std::map<std::string, Node*>& nodes) override;
};

void TransposeInverseTransposeFuser::BuildPattern() {
  // Operation patterns
  auto transpose_pattern = CreatePattern("transpose", NNADAPTER_TRANSPOSE);
  auto inverse_transpose_pattern =
      CreatePattern("inverse_transpose", NNADAPTER_TRANSPOSE)->IsIntermediate();
  // Operand patterns
  auto transpose_input_pattern =
      CreatePattern("transpose_input")
          ->IsOperationInputOperand(NNADAPTER_TRANSPOSE, 0);
  auto transpose_perm_pattern =
      CreatePattern("transpose_perm")
          ->IsOperationInputOperand(NNADAPTER_TRANSPOSE, 1);
  auto transpose_output_pattern =
      CreatePattern("transpose_output")
          ->IsOperationOutputOperand(NNADAPTER_TRANSPOSE, 0)
          ->IsOperationInputOperand(NNADAPTER_TRANSPOSE, 0)
          ->IsIntermediate();
  auto inverse_transpose_perm_pattern =
      CreatePattern("inverse_transpose_perm")
          ->IsOperationInputOperand(NNADAPTER_TRANSPOSE, 1)
          ->IsIntermediate();
  auto inverse_transpose_output_pattern =
      CreatePattern("inverse_transpose_output")
          ->IsOperationOutputOperand(NNADAPTER_TRANSPOSE, 0);
  // Create the topological connections for the above patterns
  std::vector<Pattern*> tranpose_input_patterns{transpose_input_pattern,
                                                transpose_perm_pattern};
  std::vector<Pattern*> inverse_tranpose_input_patterns{
      transpose_output_pattern, inverse_transpose_perm_pattern};
  tranpose_input_patterns >> *transpose_pattern >> *transpose_output_pattern;
  inverse_tranpose_input_patterns >> *inverse_transpose_pattern >>
      *inverse_transpose_output_pattern;
}

bool TransposeInverseTransposeFuser::HandleMatchedResults(
    core::Model* model, const std::map<std::string, Node*>& nodes) {
  // Get the operands and operations from the matched subgraph nodes.
  auto transpose_operation = nodes.at("transpose")->operation;
  auto transpose_perm_operand = nodes.at("transpose_perm")->operand;
  auto inverse_transpose_perm_operand =
      nodes.at("inverse_transpose_perm")->operand;
  auto inverse_transpose_output_operand =
      nodes.at("inverse_transpose_output")->operand;
  // Modify the permutation of the first transpose operation to an identity
  // permutation
  auto transpose_perm_count = transpose_perm_operand->length / sizeof(int32_t);
  auto transpose_perm_data =
      reinterpret_cast<int32_t*>(transpose_perm_operand->buffer);
  auto inverse_transpose_perm_count =
      inverse_transpose_perm_operand->length / sizeof(int32_t);
  auto inverse_transpose_perm_data =
      reinterpret_cast<int32_t*>(inverse_transpose_perm_operand->buffer);
  if (transpose_perm_count != inverse_transpose_perm_count) return false;
  std::vector<int32_t> transpose_perm(
      transpose_perm_data, transpose_perm_data + transpose_perm_count);
  std::vector<int32_t> inverse_transpose_perm(
      inverse_transpose_perm_data,
      inverse_transpose_perm_data + inverse_transpose_perm_count);
  if (!IsIdentityPermutation(
          MultiplyPermutation(transpose_perm, inverse_transpose_perm)))
    return false;
  // Set an identity permutation
  for (uint32_t i = 0; i < transpose_perm_count; i++) {
    transpose_perm_data[i] = i;
  }
  transpose_operation->output_operands[0] = inverse_transpose_output_operand;
  // The matched intermediate operands and operations will be deleted only when
  // it returns true.
  return true;
}

NNADAPTER_EXPORT void FuseTransposeInverseTransposeIntoIdentityTranspose(
    core::Model* model) {
  NNADAPTER_VLOG(5) << "Apply TransposeInverseTransposeFuser";
  bool stop;
  do {
    TransposeInverseTransposeFuser transpose_inverse_transpose_fuser;
    stop = transpose_inverse_transpose_fuser.Apply(model) == 0;
  } while (!stop);
}

}  // namespace nnadapter
