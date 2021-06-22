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

#include "optimizer/nchw2nhwc.h"
#include <map>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {

static const std::vector<int32_t> kNCHW2NHWC = {0, 2, 3, 1};
static const std::vector<int32_t> kNHWC2NCHW = {0, 3, 1, 2};

class NCHW2NHWCDataLayoutConverter {
 public:
  void Apply(hal::Model* model);

 private:
  void SetPermutation(hal::Operand* operand,
                      const std::vector<int32_t>& permutation) {
    NNADAPTER_CHECK(!permutation.empty());
    if (permutations_.find(operand) != permutations_.end()) {
      NNADAPTER_LOG(FATAL) << "Operand" << OperandIdToString(operand)
                           << " had been already set.";
    } else {
      permutations_[operand] = permutation;
    }
  }
  std::vector<int32_t> GetPermutation(hal::Operand* operand) {
    NNADAPTER_CHECK(permutations_.find(operand) != permutations_.end());
    return permutations_[operand];
  }

  // Operation converters
  void ConvertElementwise(hal::Operation* operation);
  void ConvertPool2D(hal::Operation* operation);
  void ConvertConcat(hal::Operation* operation);
  void ConvertConv2D(hal::Operation* operation);
  void ConvertFullyConnected(hal::Operation* operation);
  void ConvertActivation(hal::Operation* operation);
  void ConvertReshape(hal::Operation* operation);
  void ConvertSoftmax(hal::Operation* operation);
  void ConvertTranspose(hal::Operation* operation);

 private:
  hal::Model* model_{nullptr};
  std::map<hal::Operand*, std::vector<int32_t>> permutations_;
};

void NCHW2NHWCDataLayoutConverter::ConvertElementwise(
    hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 3);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto output_operand = output_operands[0];
  auto output_dimension_count = output_operand->type.dimension_count;
  // Force to align the dimorder vector of all of input operands
  std::vector<int32_t> reference_permutation;
  hal::Operand* reference_operand = nullptr;
  for (size_t i = 0; i < input_count; i++) {
    auto input_operand = input_operands[i];
    if (!IsConstantOperand(input_operand)) {
      auto input_permutation = GetPermutation(input_operand);
      if (input_permutation.size() > reference_permutation.size()) {
        reference_permutation = input_permutation;
        reference_operand = input_operand;
      }
    }
  }
  if (reference_permutation.empty()) {
    // All of input operands are constant
    SetPermutation(output_operand, IdentityPermutation(output_dimension_count));
  } else {
    auto reference_dimension_count = reference_operand->type.dimension_count;
    for (size_t i = 0; i < input_count; i++) {
      auto input_operand = input_operands[i];
      auto input_dimension_count = input_operand->type.dimension_count;
      if (!IsConstantOperand(input_operand)) {
        auto input_permutation = GetPermutation(input_operand);
        auto transpose_input_permutation = MultiplyPermutation(
            InversePermutation(input_permutation), reference_permutation);
        if (!IsIdentityPermutation(transpose_input_permutation)) {
          auto transpose_input_operand = AddTransposeOperation(
              model_, input_operand, transpose_input_permutation);
          SetPermutation(transpose_input_operand, reference_permutation);
        }
      } else {
        if (IsIdentityPermutation(reference_permutation)) {
          // Ignore
        } else if (input_dimension_count == reference_permutation.size()) {
          TransposeOperand(input_operand, reference_permutation);
        } else {
          // Expand shape with 1
          std::vector<int32_t> origin_reference_dimensions(
              reference_dimension_count);
          TransposeDimensions(reference_operand->type.dimensions,
                              InversePermutation(reference_permutation),
                              &origin_reference_dimensions[0]);
          std::vector<int32_t> expanded_input_dimensions;
          for (uint32_t j = 0, k = 0; j < reference_dimension_count; j++) {
            if (origin_reference_dimensions[j] ==
                    input_operand->type.dimensions[k] &&
                k < input_dimension_count) {
              expanded_input_dimensions.push_back(
                  input_operand->type.dimensions[k]);
              ++k;
            } else {
              expanded_input_dimensions.push_back(1);
            }
          }
        }
      }
    }
    TransposeOperand(output_operand, reference_permutation);
    SetPermutation(output_operand, reference_permutation);
  }
}

void NCHW2NHWCDataLayoutConverter::ConvertPool2D(hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 12);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  auto input_dimension_count = input_operand->type.dimension_count;
  NNADAPTER_CHECK_EQ(input_dimension_count, 4);
  auto output_operand = output_operands[0];
  // Force to apply the dimorder vector of NCHW2NHWC conversion
  auto input_permutation = GetPermutation(input_operand);
  auto transpose_input_permutation =
      MultiplyPermutation(InversePermutation(input_permutation), kNCHW2NHWC);
  if (!IsIdentityPermutation(transpose_input_permutation)) {
    auto transpose_input_operand = AddTransposeOperation(
        model_, input_operand, transpose_input_permutation);
    SetPermutation(transpose_input_operand, kNCHW2NHWC);
  }
  TransposeOperand(output_operand, kNCHW2NHWC);
  SetPermutation(output_operand, kNCHW2NHWC);
}

void NCHW2NHWCDataLayoutConverter::ConvertConcat(hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_GE(input_count, 2);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto* axis =
      reinterpret_cast<int32_t*>(input_operands[input_count - 1]->buffer);
  if (*axis < 0) {
    *axis += input_operands[0]->type.dimension_count;
  }
  auto output_operand = output_operands[0];
  auto output_dimension_count = output_operand->type.dimension_count;
  // Force to align the dimorder vector of all of input operands
  std::vector<int32_t> reference_permutation;
  hal::Operand* reference_operand = nullptr;
  for (size_t i = 0; i < input_count - 1; i++) {
    auto input_operand = input_operands[i];
    if (!IsConstantOperand(input_operand)) {
      auto input_permutation = GetPermutation(input_operand);
      if (input_permutation.size() > reference_permutation.size()) {
        reference_permutation = input_permutation;
        reference_operand = input_operand;
      }
    }
  }
  if (reference_permutation.empty()) {
    // All of input operands are constant
    SetPermutation(output_operand, IdentityPermutation(output_dimension_count));
  } else {
    for (size_t i = 0; i < input_count - 1; i++) {
      auto input_operand = input_operands[i];
      if (!IsConstantOperand(input_operand)) {
        auto input_permutation = GetPermutation(input_operand);
        auto transpose_input_permutation = MultiplyPermutation(
            InversePermutation(input_permutation), reference_permutation);
        if (!IsIdentityPermutation(transpose_input_permutation)) {
          auto transpose_input_operand = AddTransposeOperation(
              model_, input_operand, transpose_input_permutation);
          SetPermutation(transpose_input_operand, reference_permutation);
        }
      } else {
        if (IsIdentityPermutation(reference_permutation)) {
          // Ignore
        } else {
          NNADAPTER_CHECK_EQ(input_operand->type.dimension_count,
                             reference_permutation.size());
          TransposeOperand(input_operand, reference_permutation);
        }
      }
    }
    *axis = TransposeAxis(*axis, reference_permutation);
    TransposeOperand(output_operand, reference_permutation);
    SetPermutation(output_operand, reference_permutation);
  }
}

void NCHW2NHWCDataLayoutConverter::ConvertConv2D(hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 13);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  auto input_dimension_count = input_operand->type.dimension_count;
  NNADAPTER_CHECK_EQ(input_dimension_count, 4);
  auto input_channel_size = input_operand->type.dimensions[3];
  auto filter_operand = input_operands[1];
  bool is_per_channel = filter_operand->type.precision ==
                        NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL;
  NNADAPTER_VLOG(5) << "is_per_channel:" << is_per_channel;
  auto group = *reinterpret_cast<int32_t*>(input_operands[9]->buffer);
  // Check depthwise mode
  bool is_depthwise_mode = group != 1 && input_channel_size == group;
  NNADAPTER_VLOG(5) << "depthwise mode(" << is_depthwise_mode << ").";
  // Force to apply the dimorder vector of NCHW2NHWC conversion
  auto input_permutation = GetPermutation(input_operand);
  auto transpose_input_permutation =
      MultiplyPermutation(InversePermutation(input_permutation), kNCHW2NHWC);
  if (!IsIdentityPermutation(transpose_input_permutation)) {
    auto transpose_input_operand = AddTransposeOperation(
        model_, input_operand, transpose_input_permutation);
    SetPermutation(transpose_input_operand, kNCHW2NHWC);
  }
  std::vector<int32_t> filter_permutation = {};
  if (is_per_channel) {
    filter_operand->type.symm_per_channel_params.channel_dim =
        is_depthwise_mode ? 3 : 0;
  }
  if (is_depthwise_mode) {
    // [C_out, 1, filter_height, filter_width]->[1, filter_height, filter_width,
    // C_out]
    filter_permutation = {1, 2, 3, 0};
  } else {
    // [C_out, C_in, filter_height, filter_width]->[C_out, filter_height,
    // filter_width, C_in]
    filter_permutation = {0, 2, 3, 1};
  }
  TransposeOperand(filter_operand, filter_permutation);
  SetPermutation(filter_operand, filter_permutation);
  auto output_operand = output_operands[0];
  TransposeOperand(output_operand, kNCHW2NHWC);
  SetPermutation(output_operand, kNCHW2NHWC);
}

void NCHW2NHWCDataLayoutConverter::ConvertFullyConnected(
    hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 4);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  int input_dimension_count = input_operand->type.dimension_count;
  auto weight_operand = input_operands[1];
  auto bias_operand = input_operands[2];
  auto output_operand = output_operands[0];
  auto output_dimension_count = output_operand->type.dimension_count;
  // Skip NCHW2NHWC conversion
  SetPermutation(output_operand, IdentityPermutation(output_dimension_count));
}

void NCHW2NHWCDataLayoutConverter::ConvertActivation(
    hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 1);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  auto output_operand = output_operands[0];
  // The input and output operands share the same dimorder vector
  auto input_permutation = GetPermutation(input_operand);
  TransposeOperand(output_operand, input_permutation);
  SetPermutation(output_operand, input_permutation);
}

void NCHW2NHWCDataLayoutConverter::ConvertReshape(hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 2);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  int input_dimension_count = input_operand->type.dimension_count;
  auto output_operand = output_operands[0];
  auto output_dimension_count = output_operand->type.dimension_count;
  // Force to restore the dimorder vector of the input operand
  auto input_permutation = GetPermutation(input_operand);
  auto transpose_input_permutation = InversePermutation(input_permutation);
  if (!IsIdentityPermutation(transpose_input_permutation)) {
    auto transpose_input_operand = AddTransposeOperation(
        model_, input_operand, transpose_input_permutation);
    SetPermutation(transpose_input_operand,
                   IdentityPermutation(input_dimension_count));
  }
  SetPermutation(output_operand, IdentityPermutation(output_dimension_count));
}

void NCHW2NHWCDataLayoutConverter::ConvertSoftmax(hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 2);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  auto input_dimension_count = input_operand->type.dimension_count;
  auto axis = reinterpret_cast<int32_t*>(input_operands[1]->buffer);
  if (*axis < 0) {
    *axis += input_dimension_count;
  }
  auto output_operand = output_operands[0];
  // Recalculate the axis according to the dimorder vector of the input operand
  auto input_permutation = GetPermutation(input_operand);
  *axis = TransposeAxis(*axis, input_permutation);
  TransposeOperand(output_operand, input_permutation);
  SetPermutation(output_operand, input_permutation);
}

void NCHW2NHWCDataLayoutConverter::ConvertTranspose(hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 2);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  auto input_dimension_count = input_operand->type.dimension_count;
  auto perm_operand = input_operands[1];
  auto perm_count = perm_operand->length / sizeof(int32_t);
  auto perm_data = reinterpret_cast<int32_t*>(perm_operand->buffer);
  auto output_operand = output_operands[0];
  auto output_dimension_count = output_operand->type.dimension_count;
  // Recalculate the perm according to the dimorder vector of the input operand
  auto input_permutation = GetPermutation(input_operand);
  std::vector<int32_t> perm(perm_data, perm_data + perm_count);
  perm = MultiplyPermutation(
      MultiplyPermutation(InversePermutation(input_permutation), perm),
      input_permutation);
  memcpy(perm_data, &perm[0], perm_operand->length);
  TransposeOperand(output_operand, input_permutation);
  SetPermutation(output_operand, input_permutation);
}

void NCHW2NHWCDataLayoutConverter::Apply(hal::Model* model) {
  model_ = model;
  // Initialize the permutation of model input operands
  for (auto& operand : model_->input_operands) {
    SetPermutation(operand, IdentityPermutation(operand->type.dimension_count));
  }
  // Layout inference and get the permutations of all operands
  std::vector<hal::Operation*> operations =
      SortOperationsInTopologicalOrder(model_);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    switch (operation->type) {
      case NNADAPTER_ADD:
      case NNADAPTER_DIV:
      case NNADAPTER_MUL:
      case NNADAPTER_SUB:
        ConvertElementwise(operation);
        break;
      case NNADAPTER_AVERAGE_POOL_2D:
      case NNADAPTER_MAX_POOL_2D:
        ConvertPool2D(operation);
        break;
      case NNADAPTER_CONCAT:
        ConvertConcat(operation);
        break;
      case NNADAPTER_CONV_2D:
        ConvertConv2D(operation);
        break;
      case NNADAPTER_FULLY_CONNECTED:
        ConvertFullyConnected(operation);
        break;
      case NNADAPTER_RELU:
      case NNADAPTER_RELU6:
      case NNADAPTER_TANH:
      case NNADAPTER_HARD_SIGMOID:
      case NNADAPTER_HARD_SWISH:
      case NNADAPTER_SIGMOID:
        ConvertActivation(operation);
        break;
      case NNADAPTER_RESHAPE:
        ConvertReshape(operation);
        break;
      case NNADAPTER_SOFTMAX:
        ConvertSoftmax(operation);
        break;
      case NNADAPTER_TRANSPOSE:
        ConvertTranspose(operation);
        break;
      default:
        NNADAPTER_LOG(FATAL) << "Missing the processing of "
                             << OperationTypeToString(operation->type)
                             << " for the conversion from NCHW to NHWC.";
        break;
    }
  }
  // Restore all of the model output operands if they have non-identity
  // permutation
  for (auto& operand : model_->output_operands) {
    auto permutation = GetPermutation(operand);
    auto transpose_permutation = InversePermutation(permutation);
    if (!IsIdentityPermutation(transpose_permutation)) {
      auto transpose_operand =
          AddTransposeOperation(model_, operand, transpose_permutation);
      SetPermutation(transpose_operand,
                     IdentityPermutation(operand->type.dimension_count));
    }
  }
}

NNADAPTER_EXPORT void ConvertDataLayoutNCHWToNHWC(hal::Model* model) {
  NCHW2NHWCDataLayoutConverter converter;
  converter.Apply(model);
}

}  // namespace nnadapter
