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

#include "optimizer/fuse_matmul_add_into_fully_connected.h"
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

class MatMulAddFuser : public PatternMatcher {
 public:
  explicit MatMulAddFuser(bool restrict_2d_input)
      : restrict_2d_input_(restrict_2d_input) {}
  void BuildPattern() override;
  bool HandleMatchedResults(core::Model* model,
                            const std::map<std::string, Node*>& nodes) override;

 private:
  bool restrict_2d_input_{false};
};

void MatMulAddFuser::BuildPattern() {
  // Operation patterns
  auto matmul_pattern =
      CreatePattern("matmul", NNADAPTER_MAT_MUL)
          ->MatchCondition([this](const Node* node) -> bool {
            auto operation = node->operation;
            if (restrict_2d_input_) {
              return operation && operation->input_operands.size() == 4 &&
                     operation->input_operands[0]->type.dimensions.count == 2 &&
                     operation->input_operands[1]->type.dimensions.count == 2;
            } else {
              return operation && operation->input_operands.size() == 4 &&
                     operation->input_operands[0]->type.dimensions.count >= 2 &&
                     operation->input_operands[1]->type.dimensions.count == 2;
            }
          })
          ->IsIntermediate();
  auto add_pattern = CreatePattern("add", NNADAPTER_ADD)->IsIntermediate();
  // Operand patterns
  auto matmul_x_pattern = CreatePattern("matmul_x")
                              ->IsOperationInputOperand(NNADAPTER_MAT_MUL, 0)
                              ->IsVariableOperand();
  auto matmul_y_pattern = CreatePattern("matmul_y")
                              ->IsOperationInputOperand(NNADAPTER_MAT_MUL, 1)
                              ->IsConstantOperand();
  auto matmul_transpose_x_pattern =
      CreatePattern("matmul_transpose_x")
          ->IsOperationInputOperand(NNADAPTER_MAT_MUL, 2)
          ->IsConstantOperand()
          ->MatchCondition([](const Node* node) -> bool {
            auto operand = node->operand;
            return operand && operand->buffer &&
                   !*reinterpret_cast<bool*>(operand->buffer);
          })
          ->IsIntermediate();
  auto matmul_transpose_y_pattern =
      CreatePattern("matmul_transpose_y")
          ->IsOperationInputOperand(NNADAPTER_MAT_MUL, 3)
          ->IsConstantOperand()
          ->IsIntermediate();
  auto matmul_output_pattern = CreatePattern("matmul_output")->IsIntermediate();
  auto add_input_pattern = CreatePattern("add_input")
                               ->IsOperationInputOperand(NNADAPTER_ADD, 1)
                               ->IsConstantOperand()
                               ->IsIntermediate();
  auto add_fuse_code_pattern = CreatePattern("add_fuse_code")
                                   ->IsOperationInputOperand(NNADAPTER_ADD, 2)
                                   ->IsConstantOperand();
  auto add_output_pattern = CreatePattern("add_output");
  // Create the topological connections for the above patterns
  std::vector<Pattern*> matmul_input_patterns{matmul_x_pattern,
                                              matmul_y_pattern,
                                              matmul_transpose_x_pattern,
                                              matmul_transpose_y_pattern};
  std::vector<Pattern*> add_input_patterns{
      matmul_output_pattern, add_input_pattern, add_fuse_code_pattern};
  matmul_input_patterns >> *matmul_pattern >> *matmul_output_pattern;
  add_input_patterns >> *add_pattern >> *add_output_pattern;
}

bool MatMulAddFuser::HandleMatchedResults(
    core::Model* model, const std::map<std::string, Node*>& nodes) {
  // Get the operands and operations from the matched subgraph nodes.
  auto matmul_operation = nodes.at("matmul")->operation;
  auto matmul_x_operand = matmul_operation->input_operands[0];
  auto matmul_y_operand = matmul_operation->input_operands[1];
  auto matmul_transpose_y_operand = matmul_operation->input_operands[3];
  auto matmul_output_operand = matmul_operation->output_operands[0];
  auto add_operation = nodes.at("add")->operation;
  auto add_input_operand = add_operation->input_operands[1];
  auto add_fuse_code_operand = add_operation->input_operands[2];
  auto add_output_operand = add_operation->output_operands[0];
  auto matmul_transpose_y =
      *reinterpret_cast<bool*>(matmul_transpose_y_operand->buffer);
  auto matmul_num_units =
      matmul_y_operand->type.dimensions.data[matmul_transpose_y ? 0 : 1];
  if (ProductionOfDimensions(add_input_operand->type.dimensions.data,
                             add_input_operand->type.dimensions.count) !=
      matmul_num_units)
    return false;
  // [batch_size, input_size] * [input_size, num_units] -> [batch_size,
  // input_size] * [num_units, input_size]
  if (!matmul_transpose_y) {
    TransposeOperand(matmul_y_operand, std::vector<int32_t>({1, 0}));
  }
  // Requantize the input operand of NNADAPTER_ADD as the bias operand of
  // NNADAPTER_FULLY_CONNECTED
  std::vector<float> dequantized_add_input(matmul_num_units);
  if (IsInt8SymmPerLayerQuantType(add_input_operand->type.precision)) {
    NNADAPTER_CHECK(DequantizeData<int8_t>(
        reinterpret_cast<int8_t*>(add_input_operand->buffer),
        &matmul_num_units,
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
        &matmul_num_units,
        1,
        &add_input_operand->type.symm_per_layer_params.scale,
        NULL,
        -1,
        -2147483647,
        2147483647,
        dequantized_add_input.data()));
  } else {
    NNADAPTER_CHECK_EQ(add_input_operand->type.precision, NNADAPTER_FLOAT32);
    memcpy(dequantized_add_input.data(),
           reinterpret_cast<float*>(add_input_operand->buffer),
           add_input_operand->length);
  }
  core::Operand* fully_connected_bias_operand;
  // Quantize the bias operand for NNADAPTER_FULLY_CONNECTED
  if (IsInt8SymmPerLayerQuantType(matmul_x_operand->type.precision) &&
      IsInt8SymmPerLayerQuantType(matmul_y_operand->type.precision) &&
      IsInt8SymmPerLayerQuantType(matmul_output_operand->type.precision)) {
    float fully_connected_bias_scale =
        matmul_x_operand->type.symm_per_layer_params.scale *
        matmul_y_operand->type.symm_per_layer_params.scale;
    std::vector<int32_t> fully_connected_bias(matmul_num_units);
    NNADAPTER_CHECK(QuantizeData<int32_t>(dequantized_add_input.data(),
                                          &matmul_num_units,
                                          1,
                                          &fully_connected_bias_scale,
                                          NULL,
                                          -1,
                                          -2147483647,
                                          2147483647,
                                          fully_connected_bias.data()));
    fully_connected_bias_operand =
        AddQuant32ConstantOperand(model,
                                  fully_connected_bias.data(),
                                  {matmul_num_units},
                                  fully_connected_bias_scale);
  } else if (IsInt8SymmPerLayerQuantType(matmul_x_operand->type.precision) &&
             IsInt8SymmPerChannelQuantType(matmul_y_operand->type.precision) &&
             IsInt8SymmPerLayerQuantType(
                 matmul_output_operand->type.precision)) {
    std::vector<float> fully_connected_bias_scale;
    for (uint32_t i = 0; i < matmul_num_units; i++) {
      fully_connected_bias_scale.push_back(
          matmul_x_operand->type.symm_per_layer_params.scale *
          matmul_y_operand->type.symm_per_channel_params.scales[i]);
    }
    std::vector<int32_t> fully_connected_bias(matmul_num_units);
    NNADAPTER_CHECK(QuantizeData<int32_t>(dequantized_add_input.data(),
                                          &matmul_num_units,
                                          1,
                                          fully_connected_bias_scale.data(),
                                          NULL,
                                          -1,
                                          -2147483647,
                                          2147483647,
                                          fully_connected_bias.data()));
    fully_connected_bias_operand =
        AddQuant32ConstantOperand(model,
                                  fully_connected_bias.data(),
                                  {matmul_num_units},
                                  fully_connected_bias_scale.data(),
                                  matmul_num_units,
                                  0);
  } else {
    NNADAPTER_CHECK_EQ(matmul_x_operand->type.precision, NNADAPTER_FLOAT32);
    NNADAPTER_CHECK_EQ(matmul_y_operand->type.precision, NNADAPTER_FLOAT32);
    NNADAPTER_CHECK_EQ(matmul_output_operand->type.precision,
                       NNADAPTER_FLOAT32);
    fully_connected_bias_operand =
        AddFloat32ConstantOperand(model, dequantized_add_input);
  }
  // Create a new NNADAPTER_FULLY_CONNECTED operation and replace the matched
  // subgraph nodes.
  auto* fully_connected_operation = AddOperation(model);
  fully_connected_operation->type = NNADAPTER_FULLY_CONNECTED;
  fully_connected_operation->input_operands = {matmul_x_operand,
                                               matmul_y_operand,
                                               fully_connected_bias_operand,
                                               add_fuse_code_operand};
  fully_connected_operation->output_operands = {add_output_operand};
  // The matched intermediate operands and operations will be deleted only when
  // it returns true.
  return true;
}

NNADAPTER_EXPORT void FuseMatMulAddIntoFullyConnected(core::Model* model,
                                                      bool restrict_2d_input) {
  NNADAPTER_VLOG(5) << "Apply MatMulAddFuser";
  bool stop;
  do {
    MatMulAddFuser mat_mul_add_fuser(restrict_2d_input);
    stop = mat_mul_add_fuser.Apply(model) == 0;
  } while (!stop);
}

}  // namespace nnadapter
