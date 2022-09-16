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

#include "optimizer/convert_datalayout_nchw_to_nhwc.h"
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
static const std::vector<int32_t> kNCHW2CHWN = {1, 2, 3, 0};

void NCHW2NHWCDataLayoutConverter::SetPermutation(
    core::Operand* operand, const std::vector<int32_t>& permutation) {
  NNADAPTER_CHECK(!permutation.empty());
  if (permutations_.find(operand) != permutations_.end()) {
    NNADAPTER_LOG(FATAL) << "Operand" << OperandIdToString(operand)
                         << " had been already set.";
  } else {
    permutations_[operand] = permutation;
  }
}

std::vector<int32_t> NCHW2NHWCDataLayoutConverter::GetPermutation(
    core::Operand* operand) {
  NNADAPTER_CHECK(permutations_.find(operand) != permutations_.end());
  return permutations_[operand];
}

void NCHW2NHWCDataLayoutConverter::SetOperationLayout(
    core::Operation* operation, const int input_num, const int output_num) {
  for (int in_index = 0; in_index < input_num; ++in_index) {
    operation->input_operands[in_index]->type.layout =
        NNAdapterOperandLayoutCode::NNADAPTER_NHWC;
  }
  for (int out_index = 0; out_index < output_num; ++out_index) {
    operation->output_operands[out_index]->type.layout =
        NNAdapterOperandLayoutCode::NNADAPTER_NHWC;
  }
}

core::Model* NCHW2NHWCDataLayoutConverter::GetModel() { return model_; }

void NCHW2NHWCDataLayoutConverter::ConvertElementwise(
    core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 3);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto output_operand = output_operands[0];
  auto output_dimensions_count = output_operand->type.dimensions.count;
  // Force to align the dimorder vector of all of input operands
  std::vector<int32_t> reference_permutation;
  core::Operand* reference_operand = nullptr;
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
    SetPermutation(output_operand,
                   IdentityPermutation(output_dimensions_count));
  } else {
    auto reference_dimensions_count = reference_operand->type.dimensions.count;
    for (size_t i = 0; i < input_count; i++) {
      auto input_operand = input_operands[i];
      auto input_dimensions_count = input_operand->type.dimensions.count;
      if (!IsConstantOperand(input_operand)) {
        auto input_permutation = GetPermutation(input_operand);
        auto transpose_input_permutation = MultiplyPermutation(
            InversePermutation(input_permutation), reference_permutation);
        if (!IsIdentityPermutation(transpose_input_permutation)) {
          auto transpose_input_operand = AppendTransposeOperation(
              model_, input_operand, transpose_input_permutation);
          UpdateOperationInputOperands(
              {operation}, input_operand, transpose_input_operand);
          SetPermutation(transpose_input_operand, reference_permutation);
        }
      } else {
        if (IsIdentityPermutation(reference_permutation)) {
          // Ignore
        } else if (input_dimensions_count == reference_permutation.size()) {
          TransposeOperand(input_operand, reference_permutation);
        } else {
          // Expand shape with 1
          std::vector<int32_t> origin_reference_dimensions(
              reference_dimensions_count);
          TransposeDimensions(reference_operand->type.dimensions.data,
                              InversePermutation(reference_permutation),
                              &origin_reference_dimensions[0]);
          std::vector<int32_t> expanded_input_dimensions;
          for (uint32_t j = 0, k = 0; j < reference_dimensions_count; j++) {
            if (origin_reference_dimensions[j] ==
                    input_operand->type.dimensions.data[k] &&
                k < input_dimensions_count) {
              expanded_input_dimensions.push_back(
                  input_operand->type.dimensions.data[k]);
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

void NCHW2NHWCDataLayoutConverter::ConvertAdaptivePool2D(
    core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  auto operation_type = operation->type;
  if (operation_type == NNADAPTER_ADAPTIVE_AVERAGE_POOL_2D) {
    NNADAPTER_CHECK_EQ(input_count, 2);
    NNADAPTER_CHECK_EQ(output_count, 1);
  } else if (operation_type == NNADAPTER_ADAPTIVE_MAX_POOL_2D) {
    NNADAPTER_CHECK_EQ(input_count, 4);
    NNADAPTER_CHECK_EQ(output_count, 2);
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported pooling operation type "
                         << OperationTypeToString(operation->type)
                         << " is found.";
  }
  auto input_operand = input_operands[0];
  auto input_dimensions_count = input_operand->type.dimensions.count;
  NNADAPTER_CHECK_EQ(input_dimensions_count, 4);
  auto output_operand = output_operands[0];
  // Force to apply the dimorder vector of NCHW2NHWC conversion
  auto input_permutation = GetPermutation(input_operand);
  auto transpose_input_permutation =
      MultiplyPermutation(InversePermutation(input_permutation), kNCHW2NHWC);
  if (!IsIdentityPermutation(transpose_input_permutation)) {
    auto transpose_input_operand = AppendTransposeOperation(
        model_, input_operand, transpose_input_permutation);
    UpdateOperationInputOperands(
        {operation}, input_operand, transpose_input_operand);
    SetPermutation(transpose_input_operand, kNCHW2NHWC);
  }
  TransposeOperand(output_operand, kNCHW2NHWC);
  SetPermutation(output_operand, kNCHW2NHWC);
  SetOperationLayout(operation);
}

void NCHW2NHWCDataLayoutConverter::ConvertBatchNormalization(
    core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 6);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  auto output_operand = output_operands[0];
  // Force to apply the dimorder vector of NCHW2NHWC conversion
  auto input_permutation = GetPermutation(input_operand);
  auto transpose_input_permutation =
      MultiplyPermutation(InversePermutation(input_permutation), kNCHW2NHWC);
  if (!IsIdentityPermutation(transpose_input_permutation)) {
    auto transpose_input_operand = AppendTransposeOperation(
        model_, input_operand, transpose_input_permutation);
    UpdateOperationInputOperands(
        {operation}, input_operand, transpose_input_operand);
    SetPermutation(transpose_input_operand, kNCHW2NHWC);
  }
  TransposeOperand(output_operand, kNCHW2NHWC);
  SetPermutation(output_operand, kNCHW2NHWC);
  SetOperationLayout(operation);
}

void NCHW2NHWCDataLayoutConverter::ConvertPool2D(core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  auto operation_type = operation->type;
  if (operation_type == NNADAPTER_AVERAGE_POOL_2D) {
    NNADAPTER_CHECK_EQ(input_count, 8);
    NNADAPTER_CHECK_EQ(output_count, 1);
  } else if (operation_type == NNADAPTER_MAX_POOL_2D) {
    NNADAPTER_CHECK_EQ(input_count, 9);
    NNADAPTER_CHECK_EQ(output_count, 2);
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported pooling operation type "
                         << OperationTypeToString(operation->type)
                         << " is found.";
  }
  auto input_operand = input_operands[0];
  auto input_dimensions_count = input_operand->type.dimensions.count;
  NNADAPTER_CHECK_EQ(input_dimensions_count, 4);
  auto output_operand = output_operands[0];
  // Force to apply the dimorder vector of NCHW2NHWC conversion
  auto input_permutation = GetPermutation(input_operand);
  auto transpose_input_permutation =
      MultiplyPermutation(InversePermutation(input_permutation), kNCHW2NHWC);
  if (!IsIdentityPermutation(transpose_input_permutation)) {
    auto transpose_input_operand = AppendTransposeOperation(
        model_, input_operand, transpose_input_permutation);
    UpdateOperationInputOperands(
        {operation}, input_operand, transpose_input_operand);
    SetPermutation(transpose_input_operand, kNCHW2NHWC);
  }
  TransposeOperand(output_operand, kNCHW2NHWC);
  SetPermutation(output_operand, kNCHW2NHWC);
  SetOperationLayout(operation);
}

void NCHW2NHWCDataLayoutConverter::ConvertCast(core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 2);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  auto output_operand = output_operands[0];
  // The input and output operands share the same dimorder vector
  auto input_permutation = GetPermutation(input_operand);
  TransposeOperand(output_operand, input_permutation);
  SetPermutation(output_operand, input_permutation);
}

void NCHW2NHWCDataLayoutConverter::ConvertChannelShuffle(
    core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 2);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  auto output_operand = output_operands[0];
  // The input and output operands share the same dimorder vector
  auto input_permutation = GetPermutation(input_operand);
  TransposeOperand(output_operand, input_permutation);
  SetPermutation(output_operand, input_permutation);
}

void NCHW2NHWCDataLayoutConverter::ConvertClip(core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 3);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  auto output_operand = output_operands[0];
  // The input and output operands share the same dimorder vector
  auto input_permutation = GetPermutation(input_operand);
  TransposeOperand(output_operand, input_permutation);
  SetPermutation(output_operand, input_permutation);
}

void NCHW2NHWCDataLayoutConverter::ConvertConcat(core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_GE(input_count, 2);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto* axis =
      reinterpret_cast<int32_t*>(input_operands[input_count - 1]->buffer);
  if (*axis < 0) {
    *axis += input_operands[0]->type.dimensions.count;
  }
  auto output_operand = output_operands[0];
  auto output_dimensions_count = output_operand->type.dimensions.count;
  // Force to align the dimorder vector of all of input operands
  std::vector<int32_t> reference_permutation;
  core::Operand* reference_operand = nullptr;
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
    SetPermutation(output_operand,
                   IdentityPermutation(output_dimensions_count));
  } else {
    for (size_t i = 0; i < input_count - 1; i++) {
      auto input_operand = input_operands[i];
      if (!IsConstantOperand(input_operand)) {
        auto input_permutation = GetPermutation(input_operand);
        auto transpose_input_permutation = MultiplyPermutation(
            InversePermutation(input_permutation), reference_permutation);
        if (!IsIdentityPermutation(transpose_input_permutation)) {
          auto transpose_input_operand = AppendTransposeOperation(
              model_, input_operand, transpose_input_permutation);
          UpdateOperationInputOperands(
              {operation}, input_operand, transpose_input_operand);
          SetPermutation(transpose_input_operand, reference_permutation);
        }
      } else {
        if (IsIdentityPermutation(reference_permutation)) {
          // Ignore
        } else {
          NNADAPTER_CHECK_EQ(input_operand->type.dimensions.count,
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

void NCHW2NHWCDataLayoutConverter::ConvertConv2D(core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 9);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  auto input_dimensions_count = input_operand->type.dimensions.count;
  NNADAPTER_CHECK_EQ(input_dimensions_count, 4);
  auto filter_operand = input_operands[1];
  bool is_per_channel =
      filter_operand->type.precision == NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL;
  NNADAPTER_VLOG(5) << "is_per_channel:" << is_per_channel;
  auto group = *reinterpret_cast<int32_t*>(input_operands[6]->buffer);
  // Force to apply the dimorder vector of NCHW2NHWC conversion
  auto input_permutation = GetPermutation(input_operand);
  // Check depthwise mode
  auto input_channel_index = TransposeAxis(1 /* NCHW */, input_permutation);
  NNADAPTER_CHECK_GE(input_channel_index, 0);
  NNADAPTER_CHECK_LT(input_channel_index, input_dimensions_count);
  auto input_channel_size =
      input_operand->type.dimensions.data[input_channel_index];
  bool is_depthwise_mode = group != 1 && input_channel_size == group;
  NNADAPTER_VLOG(5) << "depthwise mode(" << is_depthwise_mode << ").";
  auto transpose_input_permutation =
      MultiplyPermutation(InversePermutation(input_permutation), kNCHW2NHWC);
  if (!IsIdentityPermutation(transpose_input_permutation)) {
    auto transpose_input_operand = AppendTransposeOperation(
        model_, input_operand, transpose_input_permutation);
    UpdateOperationInputOperands(
        {operation}, input_operand, transpose_input_operand);
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
  SetOperationLayout(operation, 3);
}

void NCHW2NHWCDataLayoutConverter::ConvertConv2DTranspose(
    core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 11);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  auto input_dimensions_count = input_operand->type.dimensions.count;
  NNADAPTER_CHECK_EQ(input_dimensions_count, 4);
  auto filter_operand = input_operands[1];
  bool is_per_channel =
      filter_operand->type.precision == NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL;
  NNADAPTER_VLOG(5) << "is_per_channel:" << is_per_channel;
  // Force to apply the dimorder vector of NCHW2NHWC conversion
  auto input_permutation = GetPermutation(input_operand);
  auto transpose_input_permutation =
      MultiplyPermutation(InversePermutation(input_permutation), kNCHW2NHWC);
  if (!IsIdentityPermutation(transpose_input_permutation)) {
    auto transpose_input_operand = AppendTransposeOperation(
        model_, input_operand, transpose_input_permutation);
    UpdateOperationInputOperands(
        {operation}, input_operand, transpose_input_operand);
    SetPermutation(transpose_input_operand, kNCHW2NHWC);
  }
  if (is_per_channel) {
    filter_operand->type.symm_per_channel_params.channel_dim = 0;
  }
  // [C_in, C_out, filter_height, filter_width]->[C_out, filter_height,
  // filter_width, C_in]
  TransposeOperand(filter_operand, kNCHW2CHWN);
  SetPermutation(filter_operand, kNCHW2CHWN);
  auto output_operand = output_operands[0];
  TransposeOperand(output_operand, kNCHW2NHWC);
  SetPermutation(output_operand, kNCHW2NHWC);
  SetOperationLayout(operation, 3);
}

void NCHW2NHWCDataLayoutConverter::ConvertComparisons(
    core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 2);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto output_operand = output_operands[0];
  auto output_dimensions_count = output_operand->type.dimensions.count;
  // Force to align the dimorder vector of all of input operands
  std::vector<int32_t> reference_permutation;
  core::Operand* reference_operand = nullptr;
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
    SetPermutation(output_operand,
                   IdentityPermutation(output_dimensions_count));
  } else {
    auto reference_dimensions_count = reference_operand->type.dimensions.count;
    for (size_t i = 0; i < input_count; i++) {
      auto input_operand = input_operands[i];
      auto input_dimensions_count = input_operand->type.dimensions.count;
      if (!IsConstantOperand(input_operand)) {
        auto input_permutation = GetPermutation(input_operand);
        auto transpose_input_permutation = MultiplyPermutation(
            InversePermutation(input_permutation), reference_permutation);
        if (!IsIdentityPermutation(transpose_input_permutation)) {
          auto transpose_input_operand = AppendTransposeOperation(
              model_, input_operand, transpose_input_permutation);
          UpdateOperationInputOperands(
              {operation}, input_operand, transpose_input_operand);
          SetPermutation(transpose_input_operand, reference_permutation);
        }
      } else {
        if (IsIdentityPermutation(reference_permutation)) {
          // Ignore
        } else if (input_dimensions_count == reference_permutation.size()) {
          TransposeOperand(input_operand, reference_permutation);
        } else {
          // Expand shape with 1
          std::vector<int32_t> origin_reference_dimensions(
              reference_dimensions_count);
          TransposeDimensions(reference_operand->type.dimensions.data,
                              InversePermutation(reference_permutation),
                              &origin_reference_dimensions[0]);
          std::vector<int32_t> expanded_input_dimensions;
          for (uint32_t j = 0, k = 0; j < reference_dimensions_count; j++) {
            if (origin_reference_dimensions[j] ==
                    input_operand->type.dimensions.data[k] &&
                k < input_dimensions_count) {
              expanded_input_dimensions.push_back(
                  input_operand->type.dimensions.data[k]);
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

void NCHW2NHWCDataLayoutConverter::ConvertCumSum(core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 4);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  auto input_dimensions_count = input_operand->type.dimensions.count;
  auto axis = reinterpret_cast<int32_t*>(input_operands[1]->buffer);
  if (*axis < 0) {
    *axis += input_dimensions_count;
  }
  auto output_operand = output_operands[0];
  // Recalculate the axis according to the dimorder vector of the input operand
  auto input_permutation = GetPermutation(input_operand);
  *axis = TransposeAxis(*axis, input_permutation);
  TransposeOperand(output_operand, input_permutation);
  SetPermutation(output_operand, input_permutation);
}

void NCHW2NHWCDataLayoutConverter::ConvertDequantize(
    core::Operation* operation) {
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

void NCHW2NHWCDataLayoutConverter::ConvertFullyConnected(
    core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 4);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto output_operand = output_operands[0];
  auto output_dimensions_count = output_operand->type.dimensions.count;
  // Skip NCHW2NHWC conversion
  // TODO(hong19860320) Transpose weight and output operand according to the
  // dimorder vector of the input operand
  SetPermutation(output_operand, IdentityPermutation(output_dimensions_count));
}

void NCHW2NHWCDataLayoutConverter::ConvertMatMul(core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 4);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto output_operand = output_operands[0];
  auto output_dimensions_count = output_operand->type.dimensions.count;
  // Skip NCHW2NHWC conversion
  // TODO(hong19860320) Transpose x, y and output operand, change transpose_x
  // and transpose_y according to the dimorder vector of the input operand
  SetPermutation(output_operand, IdentityPermutation(output_dimensions_count));
}

void NCHW2NHWCDataLayoutConverter::ConvertActivation(
    core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_GE(input_count, 1);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  auto output_operand = output_operands[0];
  // The input and output operands share the same dimorder vector
  auto input_permutation = GetPermutation(input_operand);
  TransposeOperand(output_operand, input_permutation);
  SetPermutation(output_operand, input_permutation);
}

void NCHW2NHWCDataLayoutConverter::ConvertLeakyRelu(
    core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 2);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  auto output_operand = output_operands[0];
  // The input and output operands share the same dimorder vector
  auto input_permutation = GetPermutation(input_operand);
  TransposeOperand(output_operand, input_permutation);
  SetPermutation(output_operand, input_permutation);
}

void NCHW2NHWCDataLayoutConverter::ConvertGelu(core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 2);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  auto output_operand = output_operands[0];
  // The input and output operands share the same dimorder vector
  auto input_permutation = GetPermutation(input_operand);
  TransposeOperand(output_operand, input_permutation);
  SetPermutation(output_operand, input_permutation);
}

void NCHW2NHWCDataLayoutConverter::ConvertGather(core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 3);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto output_operand = output_operands[0];
  auto output_dimensions_count = output_operand->type.dimensions.count;
  // Skip NCHW2NHWC conversion
  // TODO(hong19860320) Change the axis and tranpose the output operand
  // according to the dimorder vector of the input operand
  SetPermutation(output_operand, IdentityPermutation(output_dimensions_count));
}

void NCHW2NHWCDataLayoutConverter::ConvertLpNormalization(
    core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 4);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  int input_dimensions_count = input_operand->type.dimensions.count;
  auto axis = reinterpret_cast<int32_t*>(input_operands[1]->buffer);
  if (*axis < 0) {
    *axis += input_dimensions_count;
  }
  auto output_operand = output_operands[0];
  // Recalculate the axis according to the dimorder vector of the input operand
  auto input_permutation = GetPermutation(input_operand);
  *axis = TransposeAxis(*axis, input_permutation);
  // The input and output operands share the same dimorder vector
  TransposeOperand(output_operand, input_permutation);
  SetPermutation(output_operand, input_permutation);
}

void NCHW2NHWCDataLayoutConverter::ConvertPad(core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 4);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  auto output_operand = output_operands[0];
  // The input and output operands share the same dimorder vector
  auto input_permutation = GetPermutation(input_operand);
  TransposeOperand(output_operand, input_permutation);
  SetPermutation(output_operand, input_permutation);
  // Trans pads
  auto pads_data = reinterpret_cast<int32_t*>(input_operands[1]->buffer);
  int32_t pads_size = input_operands[1]->length / sizeof(int32_t);
  NNADAPTER_CHECK_EQ(input_permutation.size() * 2, pads_size);
  std::vector<int32_t> trans_pads_data(pads_size);
  for (int i = 0; i < input_permutation.size(); i++) {
    trans_pads_data[i * 2] = pads_data[input_permutation[i] * 2];
    trans_pads_data[i * 2 + 1] = pads_data[input_permutation[i] * 2 + 1];
  }
  memcpy(pads_data, trans_pads_data.data(), pads_size * sizeof(int32_t));
}

void NCHW2NHWCDataLayoutConverter::ConvertPow(core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 3);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  auto output_operand = output_operands[0];
  // The input and output operands share the same dimorder vector
  auto input_permutation = GetPermutation(input_operand);
  TransposeOperand(output_operand, input_permutation);
  SetPermutation(output_operand, input_permutation);
}

void NCHW2NHWCDataLayoutConverter::ConvertQuantize(core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 4);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  auto output_operand = output_operands[0];
  // The input and output operands share the same dimorder vector
  auto input_permutation = GetPermutation(input_operand);
  TransposeOperand(output_operand, input_permutation);
  SetPermutation(output_operand, input_permutation);
}

void NCHW2NHWCDataLayoutConverter::ConvertReduce(core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 3);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  int input_dimensions_count = input_operand->type.dimensions.count;
  auto axis = reinterpret_cast<int32_t*>(input_operands[1]->buffer);
  if (*axis < 0) {
    *axis += input_dimensions_count;
  }
  auto output_operand = output_operands[0];
  // Recalculate the axis according to the dimorder vector of the input operand
  auto input_permutation = GetPermutation(input_operand);
  *axis = TransposeAxis(*axis, input_permutation);
  auto output_dimensions_count = output_operand->type.dimensions.count;
  // Skip NCHW2NHWC conversion
  // TODO(hong19860320) Transpose output operand according to the dimorder
  // vector of the input operand
  SetPermutation(output_operand, IdentityPermutation(output_dimensions_count));
}

void NCHW2NHWCDataLayoutConverter::ConvertReshape(core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 2);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  int input_dimensions_count = input_operand->type.dimensions.count;
  auto output_operand = output_operands[0];
  auto output_dimensions_count = output_operand->type.dimensions.count;
  // Force to restore the dimorder vector of the input operand
  auto input_permutation = GetPermutation(input_operand);
  auto transpose_input_permutation = InversePermutation(input_permutation);
  if (!IsIdentityPermutation(transpose_input_permutation)) {
    auto transpose_input_operand = AppendTransposeOperation(
        model_, input_operand, transpose_input_permutation);
    UpdateOperationInputOperands(
        {operation}, input_operand, transpose_input_operand);
    SetPermutation(transpose_input_operand,
                   IdentityPermutation(input_dimensions_count));
  }
  SetPermutation(output_operand, IdentityPermutation(output_dimensions_count));
}

void NCHW2NHWCDataLayoutConverter::ConvertResizeLinear(
    core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 5);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  auto output_operand = output_operands[0];
  // Force to apply the dimorder vector of NCHW2NHWC conversion
  auto input_permutation = GetPermutation(input_operand);
  auto transpose_input_permutation =
      MultiplyPermutation(InversePermutation(input_permutation), kNCHW2NHWC);
  if (!IsIdentityPermutation(transpose_input_permutation)) {
    auto transpose_input_operand = AppendTransposeOperation(
        model_, input_operand, transpose_input_permutation);
    UpdateOperationInputOperands(
        {operation}, input_operand, transpose_input_operand);
    SetPermutation(transpose_input_operand, kNCHW2NHWC);
  }
  TransposeOperand(output_operand, kNCHW2NHWC);
  SetPermutation(output_operand, kNCHW2NHWC);
  SetOperationLayout(operation);
}

void NCHW2NHWCDataLayoutConverter::ConvertResizeNearest(
    core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 4);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  auto output_operand = output_operands[0];
  // Force to apply the dimorder vector of NCHW2NHWC conversion
  auto input_permutation = GetPermutation(input_operand);
  auto transpose_input_permutation =
      MultiplyPermutation(InversePermutation(input_permutation), kNCHW2NHWC);
  if (!IsIdentityPermutation(transpose_input_permutation)) {
    auto transpose_input_operand = AppendTransposeOperation(
        model_, input_operand, transpose_input_permutation);
    UpdateOperationInputOperands(
        {operation}, input_operand, transpose_input_operand);
    SetPermutation(transpose_input_operand, kNCHW2NHWC);
  }
  TransposeOperand(output_operand, kNCHW2NHWC);
  SetPermutation(output_operand, kNCHW2NHWC);
  SetOperationLayout(operation);
}

void NCHW2NHWCDataLayoutConverter::ConvertShape(core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 2);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto output_operand = output_operands[0];
  auto output_dimensions_count = output_operand->type.dimensions.count;
  // Skip NCHW2NHWC conversion
  SetPermutation(output_operand, IdentityPermutation(output_dimensions_count));
}

void NCHW2NHWCDataLayoutConverter::ConvertSlice(core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 5);
  NNADAPTER_CHECK_EQ(output_count, 1);
  // Recalculate the axis according to the dimorder vector of the input operand
  auto axes_operand = input_operands[1];
  int axes_count = axes_operand->length / sizeof(int32_t);
  int* axes = reinterpret_cast<int32_t*>(axes_operand->buffer);
  auto input_permutation = GetPermutation(input_operands[0]);
  for (int i = 0; i < axes_count; i++) {
    axes[i] = TransposeAxis(axes[i], input_permutation);
  }
  auto output_operand = output_operands[0];
  TransposeOperand(output_operand, input_permutation);
  SetPermutation(output_operand, input_permutation);
}

void NCHW2NHWCDataLayoutConverter::ConvertFill(core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 2);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto output_operand = output_operands[0];
  auto output_dimensions_count = output_operand->type.dimensions.count;
  // Skip NCHW2NHWC conversion
  SetPermutation(output_operand, IdentityPermutation(output_dimensions_count));
}

void NCHW2NHWCDataLayoutConverter::ConvertFillLike(core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 2);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto output_operand = output_operands[0];
  auto output_dimensions_count = output_operand->type.dimensions.count;
  // Skip NCHW2NHWC conversion
  SetPermutation(output_operand, IdentityPermutation(output_dimensions_count));
}

void NCHW2NHWCDataLayoutConverter::ConvertFlatten(core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 3);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  auto input_dimensions_count = input_operand->type.dimensions.count;
  auto output_operand = output_operands[0];
  auto output_dimensions_count = output_operand->type.dimensions.count;
  // Force to restore the dimorder vector of the input operand
  auto input_permutation = GetPermutation(input_operand);
  auto transpose_input_permutation = InversePermutation(input_permutation);
  if (!IsIdentityPermutation(transpose_input_permutation)) {
    auto transpose_input_operand = AppendTransposeOperation(
        model_, input_operand, transpose_input_permutation);
    UpdateOperationInputOperands(
        {operation}, input_operand, transpose_input_operand);
    SetPermutation(transpose_input_operand,
                   IdentityPermutation(input_dimensions_count));
  }
  SetPermutation(output_operand, IdentityPermutation(output_dimensions_count));
}

void NCHW2NHWCDataLayoutConverter::ConvertLayerNormalization(
    core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 5);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  auto input_dimensions_count = input_operand->type.dimensions.count;
  auto output_operand = output_operands[0];
  auto output_dimensions_count = output_operand->type.dimensions.count;
  // Force to restore the dimorder vector of the input operand
  auto input_permutation = GetPermutation(input_operand);
  auto transpose_input_permutation = InversePermutation(input_permutation);
  if (!IsIdentityPermutation(transpose_input_permutation)) {
    auto transpose_input_operand = AppendTransposeOperation(
        model_, input_operand, transpose_input_permutation);
    UpdateOperationInputOperands(
        {operation}, input_operand, transpose_input_operand);
    SetPermutation(transpose_input_operand,
                   IdentityPermutation(input_dimensions_count));
  }
  SetPermutation(output_operand, IdentityPermutation(output_dimensions_count));
}

void NCHW2NHWCDataLayoutConverter::ConvertSoftmax(core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 2);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  auto input_dimensions_count = input_operand->type.dimensions.count;
  auto axis = reinterpret_cast<int32_t*>(input_operands[1]->buffer);
  if (*axis < 0) {
    *axis += input_dimensions_count;
  }
  auto output_operand = output_operands[0];
  // Recalculate the axis according to the dimorder vector of the input operand
  auto input_permutation = GetPermutation(input_operand);
  *axis = TransposeAxis(*axis, input_permutation);
  TransposeOperand(output_operand, input_permutation);
  SetPermutation(output_operand, input_permutation);
}

void NCHW2NHWCDataLayoutConverter::ConvertSqueeze(core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 2);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  auto input_dimensions_count = input_operand->type.dimensions.count;
  auto output_operand = output_operands[0];
  auto output_dimensions_count = output_operand->type.dimensions.count;
  auto axes_operand = input_operands[1];
  // Recalculate the perm according to the dimorder vector of the input operand
  auto input_permutation = GetPermutation(input_operand);
  std::vector<int32_t> identity_permutation =
      IdentityPermutation(input_dimensions_count);
  std::vector<int32_t> axes;
  if (axes_operand && (axes_operand->length / sizeof(int32_t)) > 0) {
    auto axes_count = axes_operand->length / sizeof(int32_t);
    auto axes_data = reinterpret_cast<int32_t*>(axes_operand->buffer);
    // Recalculate the axes according to the dimorder vector of the input
    // operand
    for (int32_t i = 0; i < axes_count; i++) {
      if (axes_data[i] < 0) {
        axes_data[i] += input_dimensions_count;
      }
      // Delete the dimension corresponding to the axis of the
      // identity_permutation
      for (auto it = identity_permutation.begin();
           it != identity_permutation.end();) {
        if (*it == axes_data[i]) {
          it = identity_permutation.erase(it);
        } else {
          ++it;
        }
      }
      // Delete the dimension corresponding to the axis of the input_permutation
      TransposeAxis(axes_data[i], input_permutation);
      for (auto it = input_permutation.begin();
           it != input_permutation.end();) {
        if (*it == axes_data[i]) {
          it = input_permutation.erase(it);
        } else {
          ++it;
        }
      }
    }
    // Calculate the distance between current data layout and origin data layout
    std::vector<int32_t> output_permutation;
    for (auto identity_data : identity_permutation) {
      int32_t index = std::distance(input_permutation.begin(),
                                    std::find(input_permutation.begin(),
                                              input_permutation.end(),
                                              identity_data));
      output_permutation.push_back(index);
    }
    TransposeOperand(output_operand, output_permutation);
    SetPermutation(output_operand, output_permutation);
  } else {
    // Force to restore the dimorder vector of the input operand
    auto input_permutation = GetPermutation(input_operand);
    auto transpose_input_permutation = InversePermutation(input_permutation);
    if (!IsIdentityPermutation(transpose_input_permutation)) {
      auto transpose_input_operand = AppendTransposeOperation(
          model_, input_operand, transpose_input_permutation);
      UpdateOperationInputOperands(
          {operation}, input_operand, transpose_input_operand);
      SetPermutation(transpose_input_operand,
                     IdentityPermutation(input_dimensions_count));
    }
    SetPermutation(output_operand,
                   IdentityPermutation(output_dimensions_count));
  }
}

void NCHW2NHWCDataLayoutConverter::ConvertSplit(core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 3);
  NNADAPTER_CHECK_GE(output_count, 1);
  auto input_operand = input_operands[0];
  auto input_dimensions_count = input_operand->type.dimensions.count;
  auto axis = reinterpret_cast<int32_t*>(input_operands[1]->buffer);
  if (*axis < 0) {
    *axis += input_dimensions_count;
  }
  // Recalculate the axis according to the dimorder vector of the input operand
  auto input_permutation = GetPermutation(input_operand);
  *axis = TransposeAxis(*axis, input_permutation);
  for (size_t i = 0; i < output_count; i++) {
    auto output_operand = output_operands[i];
    TransposeOperand(output_operand, input_permutation);
    SetPermutation(output_operand, input_permutation);
  }
}

void NCHW2NHWCDataLayoutConverter::ConvertUnstack(core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 3);
  NNADAPTER_CHECK_GE(output_count, 1);
  auto input_operand = input_operands[0];
  auto input_dimensions_count = input_operand->type.dimensions.count;
  auto axis = reinterpret_cast<int32_t*>(input_operands[1]->buffer);
  if (*axis < 0) {
    *axis += input_dimensions_count;
  }
  /* Example:                                                        */
  /*  - Format: NCHW                      Format: NHWC
      - Dims: {6, 7, 8, 4}       ->       Dims: {6, 8, 4, 7}
      - Axis : 2                          Aixs: 1
      - Output: 8 * {6, 7, 4}             Output: 8 * {6, 4, 7} */
  /* Src format NCHW: Identity permutation: {0, 1, 2, 3}, aixs = 2, output
   * identity permutation: {0, 1, 3} */
  /* Dst format NHWC: Input permutation: {0, 2, 3, 1}, axis = 1, output
   * permutation: {0, 3, 1} */
  /* We need trans output identity permutation {0, 1, 3} with format NCHW to
   * output permutation {0, 3, 1} with format NHWC. */
  /* Set output permutation: {0, 2, 1} */
  std::vector<int32_t> input_identity_permutation =
      IdentityPermutation(input_dimensions_count);
  input_identity_permutation.erase(input_identity_permutation.begin() + *axis);
  auto input_permutation = GetPermutation(input_operand);
  *axis = TransposeAxis(*axis, input_permutation);
  input_permutation.erase(input_permutation.begin() + *axis);
  NNADAPTER_CHECK_EQ(input_identity_permutation.size(),
                     input_permutation.size());
  std::vector<int32_t> output_permutation{};
  for (int i = 0; i < input_permutation.size(); i++) {
    for (int j = 0; j < input_identity_permutation.size(); j++) {
      if (input_identity_permutation[j] == input_permutation[i]) {
        output_permutation.push_back(i);
        break;
      }
    }
  }
  bool is_need_transpose = !IsIdentityPermutation(output_permutation);
  for (size_t i = 0; i < output_count; i++) {
    auto output_operand = output_operands[i];
    if (is_need_transpose) {
      TransposeOperand(output_operand, output_permutation);
    }
    SetPermutation(output_operand, output_permutation);
  }
}

void NCHW2NHWCDataLayoutConverter::ConvertStack(core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_GE(input_count, 2);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto* axis =
      reinterpret_cast<int32_t*>(input_operands[input_count - 1]->buffer);
  if (*axis < 0) {
    *axis += input_operands[0]->type.dimensions.count;
  }
  auto output_operand = output_operands[0];
  auto output_dimensions_count = output_operand->type.dimensions.count;
  // Force to align the dimorder vector of all of input operands
  std::vector<int32_t> reference_permutation;
  core::Operand* reference_operand = nullptr;
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
    SetPermutation(output_operand,
                   IdentityPermutation(output_dimensions_count));
  } else {
    for (size_t i = 0; i < input_count - 1; i++) {
      auto input_operand = input_operands[i];
      if (!IsConstantOperand(input_operand)) {
        auto input_permutation = GetPermutation(input_operand);
        auto transpose_input_permutation = MultiplyPermutation(
            InversePermutation(input_permutation), reference_permutation);
        if (!IsIdentityPermutation(transpose_input_permutation)) {
          auto transpose_input_operand = AppendTransposeOperation(
              model_, input_operand, transpose_input_permutation);
          UpdateOperationInputOperands(
              {operation}, input_operand, transpose_input_operand);
          SetPermutation(transpose_input_operand, reference_permutation);
        }
      } else {
        if (IsIdentityPermutation(reference_permutation)) {
          // Ignore
        } else {
          NNADAPTER_CHECK_EQ(input_operand->type.dimensions.count,
                             reference_permutation.size());
          TransposeOperand(input_operand, reference_permutation);
        }
      }
    }
    for (auto& perm_data : reference_permutation) {
      if (perm_data >= *axis) perm_data += 1;
    }
    reference_permutation.insert(reference_permutation.begin() + *axis, *axis);
    TransposeOperand(output_operand, reference_permutation);
    SetPermutation(output_operand, reference_permutation);
  }
}

void NCHW2NHWCDataLayoutConverter::ConvertTile(core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 2);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  auto output_operand = output_operands[0];
  // The input and output operands share the same dimorder vector
  NNADAPTER_CHECK(IsConstantOperand(input_operands[1]));
  int32_t* repeat_data = reinterpret_cast<int32_t*>(input_operands[1]->buffer);
  int32_t repeat_count = input_operands[1]->length / sizeof(int32_t);
  std::vector<int32_t> repeat(repeat_count);
  auto input_permutation = GetPermutation(input_operand);
  NNADAPTER_CHECK_EQ(repeat_count,
                     static_cast<int32_t>(input_permutation.size()));
  for (int i = 0; i < repeat_count; i++) {
    repeat[i] = repeat_data[input_permutation[i]];
  }
  memcpy(repeat_data, repeat.data(), input_operands[1]->length);
  TransposeOperand(output_operand, input_permutation);
  SetPermutation(output_operand, input_permutation);
}

void NCHW2NHWCDataLayoutConverter::ConvertTranspose(
    core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 2);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  auto perm_operand = input_operands[1];
  auto perm_count = perm_operand->length / sizeof(int32_t);
  auto perm_data = reinterpret_cast<int32_t*>(perm_operand->buffer);
  auto output_operand = output_operands[0];
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

void NCHW2NHWCDataLayoutConverter::ConvertUnsqueeze(
    core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 2);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  int input_dimensions_count = input_operand->type.dimensions.count;
  auto output_operand = output_operands[0];
  auto output_dimensions_count = output_operand->type.dimensions.count;
  // Force to restore the dimorder vector of the input operand
  auto input_permutation = GetPermutation(input_operand);
  auto transpose_input_permutation = InversePermutation(input_permutation);
  if (!IsIdentityPermutation(transpose_input_permutation)) {
    auto transpose_input_operand = AppendTransposeOperation(
        model_, input_operand, transpose_input_permutation);
    UpdateOperationInputOperands(
        {operation}, input_operand, transpose_input_operand);
    SetPermutation(transpose_input_operand,
                   IdentityPermutation(input_dimensions_count));
  }
  SetPermutation(output_operand, IdentityPermutation(output_dimensions_count));
}

void NCHW2NHWCDataLayoutConverter::Apply(core::Model* model) {
  model_ = model;
  // Initialize the permutation of model input operands
  for (auto& operand : model_->input_operands) {
    SetPermutation(operand,
                   IdentityPermutation(operand->type.dimensions.count));
  }
  // Layout inference and get the permutations of all operands
  std::vector<core::Operation*> operations =
      SortOperationsInTopologicalOrder(model_);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    switch (operation->type) {
      case NNADAPTER_ADD:
      case NNADAPTER_DIV:
      case NNADAPTER_MAX:
      case NNADAPTER_MIN:
      case NNADAPTER_MUL:
      case NNADAPTER_SUB:
        ConvertElementwise(operation);
        break;
      case NNADAPTER_ADAPTIVE_AVERAGE_POOL_2D:
        ConvertAdaptivePool2D(operation);
        break;
      case NNADAPTER_AVERAGE_POOL_2D:
      case NNADAPTER_MAX_POOL_2D:
        ConvertPool2D(operation);
        break;
      case NNADAPTER_BATCH_NORMALIZATION:
        ConvertBatchNormalization(operation);
        break;
      case NNADAPTER_CAST:
        ConvertCast(operation);
        break;
      case NNADAPTER_CHANNEL_SHUFFLE:
        ConvertChannelShuffle(operation);
        break;
      case NNADAPTER_CLIP:
        ConvertClip(operation);
        break;
      case NNADAPTER_CONCAT:
        ConvertConcat(operation);
        break;
      case NNADAPTER_CONV_2D:
        ConvertConv2D(operation);
        break;
      case NNADAPTER_CONV_2D_TRANSPOSE:
        ConvertConv2DTranspose(operation);
        break;
      case NNADAPTER_CUM_SUM:
        ConvertCumSum(operation);
        break;
      case NNADAPTER_DEQUANTIZE:
        ConvertDequantize(operation);
        break;
      case NNADAPTER_EQUAL:
      case NNADAPTER_GREATER:
      case NNADAPTER_GREATER_EQUAL:
      case NNADAPTER_LESS:
      case NNADAPTER_LESS_EQUAL:
      case NNADAPTER_NOT_EQUAL:
        ConvertComparisons(operation);
        break;
      case NNADAPTER_FILL:
        ConvertFill(operation);
        break;
      case NNADAPTER_FILL_LIKE:
        ConvertFillLike(operation);
        break;
      case NNADAPTER_FLATTEN:
        ConvertFlatten(operation);
        break;
      case NNADAPTER_FULLY_CONNECTED:
        ConvertFullyConnected(operation);
        break;
      case NNADAPTER_GATHER:
        ConvertGather(operation);
        break;
      case NNADAPTER_GELU:
        ConvertGelu(operation);
        break;
      case NNADAPTER_LAYER_NORMALIZATION:
        ConvertLayerNormalization(operation);
        break;
      case NNADAPTER_LP_NORMALIZATION:
        ConvertLpNormalization(operation);
        break;
      case NNADAPTER_LEAKY_RELU:
        ConvertLeakyRelu(operation);
        break;
      case NNADAPTER_MAT_MUL:
        ConvertMatMul(operation);
        break;
      case NNADAPTER_PAD:
        ConvertPad(operation);
        break;
      case NNADAPTER_POW:
        ConvertPow(operation);
        break;
      case NNADAPTER_QUANTIZE:
        ConvertQuantize(operation);
        break;
      case NNADAPTER_REDUCE_MEAN:
      case NNADAPTER_REDUCE_SUM:
        ConvertReduce(operation);
        break;
      case NNADAPTER_RESIZE_NEAREST:
        ConvertResizeNearest(operation);
        break;
      case NNADAPTER_RESIZE_LINEAR:
        ConvertResizeLinear(operation);
        break;
      case NNADAPTER_RELU:
      case NNADAPTER_RELU6:
      case NNADAPTER_TANH:
      case NNADAPTER_HARD_SIGMOID:
      case NNADAPTER_HARD_SWISH:
      case NNADAPTER_SIGMOID:
      case NNADAPTER_ABS:
      case NNADAPTER_EXP:
      case NNADAPTER_LOG:
      case NNADAPTER_SWISH:
        ConvertActivation(operation);
        break;
      case NNADAPTER_RESHAPE:
        ConvertReshape(operation);
        break;
      case NNADAPTER_SHAPE:
        ConvertShape(operation);
        break;
      case NNADAPTER_SLICE:
        ConvertSlice(operation);
        break;
      case NNADAPTER_SOFTMAX:
        ConvertSoftmax(operation);
        break;
      case NNADAPTER_SQUEEZE:
        ConvertSqueeze(operation);
        break;
      case NNADAPTER_SPLIT:
        ConvertSplit(operation);
        break;
      case NNADAPTER_STACK:
        ConvertStack(operation);
        break;
      case NNADAPTER_TILE:
        ConvertTile(operation);
        break;
      case NNADAPTER_TRANSPOSE:
        ConvertTranspose(operation);
        break;
      case NNADAPTER_UNSTACK:
        ConvertUnstack(operation);
        break;
      case NNADAPTER_UNSQUEEZE:
        ConvertUnsqueeze(operation);
        break;
      default:
        NNADAPTER_LOG(FATAL)
            << "Missing the processing of "
            << OperationTypeToString(operation->type)
            << " for the conversion from NCHW data layout to NHWC data layout.";
        break;
    }
  }
  // Restore all of the model output operands if they have non-identity
  // permutation
  for (auto& output_operand : model_->output_operands) {
    auto output_permutation = GetPermutation(output_operand);
    auto transpose_output_permutation = InversePermutation(output_permutation);
    if (!IsIdentityPermutation(transpose_output_permutation)) {
      auto transpose_output_operand = AppendTransposeOperation(
          model_, output_operand, transpose_output_permutation);
      SetPermutation(
          transpose_output_operand,
          IdentityPermutation(output_operand->type.dimensions.count));
      // Update the current model output operand to the new operand
      output_operand->type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;
      transpose_output_operand->type.lifetime = NNADAPTER_MODEL_OUTPUT;
      output_operand = transpose_output_operand;
    }
  }
}

NNADAPTER_EXPORT void ConvertDataLayoutNCHWToNHWC(core::Model* model) {
  NCHW2NHWCDataLayoutConverter converter;
  converter.Apply(model);
}

}  // namespace nnadapter
