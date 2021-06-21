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

#include "utility/utility.h"
#include "utility/debug.h"
#include "utility/micros.h"

namespace nnadapter {

NNADAPTER_EXPORT bool IsPerLayerQuantization(
    NNAdapterOperandPrecisionCode type) {
  return type == NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER ||
         type == NNADAPTER_TENSOR_QUANT_UINT8_ASYMM_PER_LAYER ||
         type == NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_LAYER ||
         type == NNADAPTER_TENSOR_QUANT_UINT32_ASYMM_PER_LAYER;
}

NNADAPTER_EXPORT bool IsPerChannelQuantization(
    NNAdapterOperandPrecisionCode type) {
  return type == NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL ||
         type == NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_CHANNEL;
}

NNADAPTER_EXPORT bool IsAsymmetricQuantization(
    NNAdapterOperandPrecisionCode type) {
  return type == NNADAPTER_TENSOR_QUANT_UINT8_ASYMM_PER_LAYER ||
         type == NNADAPTER_TENSOR_QUANT_UINT32_ASYMM_PER_LAYER;
}

NNADAPTER_EXPORT bool IsSymmetricQuantization(
    NNAdapterOperandPrecisionCode type) {
  return type == NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER ||
         type == NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL ||
         type == NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_LAYER ||
         type == NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_CHANNEL;
}

NNADAPTER_EXPORT bool IsAsymmPerLayerQuantization(
    NNAdapterOperandPrecisionCode type) {
  return IsAsymmetricQuantization(type) && IsPerLayerQuantization(type);
}

NNADAPTER_EXPORT bool IsSymmPerLayerQuantization(
    NNAdapterOperandPrecisionCode type) {
  return IsSymmetricQuantization(type) && IsPerLayerQuantization(type);
}

NNADAPTER_EXPORT bool IsSymmPerChannelQuantization(
    NNAdapterOperandPrecisionCode type) {
  return IsSymmetricQuantization(type) && IsPerChannelQuantization(type);
}

NNADAPTER_EXPORT bool IsUInt8AsymmPerLayerQuantization(
    NNAdapterOperandPrecisionCode type) {
  return type == NNADAPTER_TENSOR_QUANT_UINT8_ASYMM_PER_LAYER;
}

NNADAPTER_EXPORT bool IsInt8SymmPerLayerQuantization(
    NNAdapterOperandPrecisionCode type) {
  return type == NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER;
}

NNADAPTER_EXPORT bool IsInt8SymmPerChannelQuantization(
    NNAdapterOperandPrecisionCode type) {
  return type == NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL;
}

NNADAPTER_EXPORT bool IsUInt32AsymmPerLayerQuantization(
    NNAdapterOperandPrecisionCode type) {
  return type == NNADAPTER_TENSOR_QUANT_UINT32_ASYMM_PER_LAYER;
}

NNADAPTER_EXPORT bool IsInt32SymmPerLayerQuantization(
    NNAdapterOperandPrecisionCode type) {
  return type == NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_LAYER;
}

NNADAPTER_EXPORT bool IsInt32SymmPerChannelQuantization(
    NNAdapterOperandPrecisionCode type) {
  return type == NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_CHANNEL;
}

NNADAPTER_EXPORT int64_t ProductionOfDimensions(
    const int32_t* input_dimensions, uint32_t input_dimension_count) {
  int64_t production = 1;
  for (uint32_t i = 0; i < input_dimension_count; i++) {
    auto dimension = input_dimensions[i];
    NNADAPTER_CHECK_GT(dimension, 0);
    production *= dimension;
  }
  return production;
}

NNADAPTER_EXPORT int64_t
ProductionOfDimensions(const std::vector<int32_t>& input_dimensions) {
  return !input_dimensions.empty()
             ? ProductionOfDimensions(&input_dimensions[0],
                                      input_dimensions.size())
             : 1;
}

NNADAPTER_EXPORT void TransposeDimensions(
    int32_t* input_dimensions,
    const std::vector<int32_t>& permutation,
    int32_t* output_dimensions_ptr) {
  size_t permutation_count = permutation.size();
  std::vector<int32_t> origin_dimensions(input_dimensions,
                                         input_dimensions + permutation_count);
  for (size_t i = 0; i < permutation_count; i++) {
    auto dimension = origin_dimensions[permutation[i]];
    if (output_dimensions_ptr != nullptr) {
      output_dimensions_ptr[i] = dimension;
    } else {
      input_dimensions[i] = dimension;
    }
  }
}

NNADAPTER_EXPORT std::vector<int32_t> IdentityPermutation(size_t rank) {
  std::vector<int32_t> permutation(rank);
  for (size_t i = 0; i < rank; i++) {
    permutation[i] = i;
  }
  return permutation;
}

NNADAPTER_EXPORT std::vector<int32_t> InversePermutation(
    const std::vector<int32_t>& permutation) {
  auto rank = permutation.size();
  std::vector<int32_t> inverse_permutation(rank);
  for (size_t i = 0; i < rank; i++) {
    inverse_permutation[permutation[i]] = i;
  }
  return inverse_permutation;
}

NNADAPTER_EXPORT std::vector<int32_t> MutiplyPermutation(
    const std::vector<int32_t>& permutation,
    const std::vector<int32_t>& multiplier) {
  auto rank = permutation.size();
  std::vector<int32_t> multiply_permutation(rank);
  for (size_t i = 0; i < multiplier.size(); i++) {
    multiply_permutation[i] = permutation[multiplier[i]];
  }
  return multiply_permutation;
}

NNADAPTER_EXPORT bool IsIdentityPermutation(
    const std::vector<int32_t>& permutation) {
  auto rank = permutation.size();
  for (size_t i = 0; i < rank; i++) {
    if (permutation[i] != i) return false;
  }
  return true;
}

NNADAPTER_EXPORT void ReshapeDimensions(int32_t* input_dimensions,
                                        uint32_t* input_dimension_count,
                                        const std::vector<int32_t>& dimensions,
                                        int32_t* output_dimensions_ptr,
                                        uint32_t* output_dimension_count_ptr) {
  const int64_t input_size =
      ProductionOfDimensions(input_dimensions, *input_dimension_count);
  bool all_positive = std::all_of(
      dimensions.cbegin(), dimensions.cend(), [](int32_t i) { return i > 0; });
  // Only one dimension can be set to -1, whose size will be automatically
  // infered.
  const int32_t unk_dim_val = -1;
  const int32_t copy_dim_val = 0;
  std::vector<int32_t> output_dimensions(dimensions.size(), 0);
  int64_t capacity = 1;
  int unk_dim_idx = -1;
  for (size_t i = 0; i < dimensions.size(); ++i) {
    if (dimensions[i] == unk_dim_val) {
      NNADAPTER_CHECK_EQ(unk_dim_idx, -1)
          << "Only one dimension value of 'dimensions' can be -1. But received "
             "new_dimensions = ["
          << DimensionsToString(&dimensions[0], dimensions.size())
          << "], dimensions[" << i << "] is also -1.";
      unk_dim_idx = i;
    } else if (dimensions[i] == copy_dim_val) {
      NNADAPTER_CHECK_LT(static_cast<uint32_t>(i), *input_dimension_count)
          << "The index of 0 in `dimensions` must be less than the "
             "input_dimension_count. But received dimensions = ["
          << DimensionsToString(&dimensions[0], dimensions.size())
          << "], dimensions[" << i << "] = 0, input_dimensions = ["
          << DimensionsToString(input_dimensions, *input_dimension_count)
          << "], input_dimension_count = " << *input_dimension_count << ".";
    } else {
      NNADAPTER_CHECK_GT(dimensions[i], 0)
          << "Each dimension value of 'dimensions' must not be negative except "
             "one unknown dimension. But received dimensions = ["
          << DimensionsToString(&dimensions[0], dimensions.size())
          << "], dimensions[" << i << "] = " << dimensions[i] << ".";
    }
    capacity *= (dimensions[i] ? dimensions[i] : input_dimensions[i]);
    output_dimensions[i] = (dimensions[i] ? static_cast<int32_t>(dimensions[i])
                                          : input_dimensions[i]);
  }
  if (unk_dim_idx != -1) {
    if (all_positive) {
      // input_size < 0 and is un-determinate in compile time, skip the check,
      // for example, input_dimensions = [-1, 8, 1, 1], dimensions = [-1, 3, 8],
      // capacity = -24, input_size = -8, output_dimensions[0] = 0 the following
      // check will fail.
      output_dimensions[unk_dim_idx] = -input_size / capacity;
      NNADAPTER_CHECK_EQ(output_dimensions[unk_dim_idx] * capacity, -input_size)
          << "The 'dimensions' is invalid. The input size must be divisible by "
             "known capacity of 'dimensions'. But received input_dimensions = ["
          << DimensionsToString(input_dimensions, *input_dimension_count)
          << "], input size = " << input_size << ", 'dimensions' is ["
          << DimensionsToString(&dimensions[0], dimensions.size())
          << "], known capacity of 'dimensions' is " << capacity << ".";
    } else {
      output_dimensions[unk_dim_idx] = -1;
    }
  } else {
    if (all_positive) {
      NNADAPTER_CHECK_EQ(capacity, input_size)
          << "The 'dimensions' is invalid. The input size must be equal to the "
             "capacity of 'dimensions'. But received input_dimensions = ["
          << DimensionsToString(input_dimensions, *input_dimension_count)
          << "], input size = " << input_size << ", 'dimensions' is ["
          << DimensionsToString(&dimensions[0], dimensions.size())
          << "], the capacity of 'dimensions' is " << capacity << ".";
    }
  }
  if (!output_dimensions_ptr || !output_dimension_count_ptr) {
    output_dimensions_ptr = input_dimensions;
    output_dimension_count_ptr = input_dimension_count;
  }
  *output_dimension_count_ptr = output_dimensions.size();
  memcpy(output_dimensions_ptr,
         &output_dimensions[0],
         output_dimensions.size() * sizeof(int32_t));
}

NNADAPTER_EXPORT void TransposeAxis(int32_t axis,
                                    const std::vector<int32_t>& permutation) {
  NNADAPTER_CHECK_GE(axis, 0);
  int32_t new_axis = -1;
  for (size_t i = 0; i < permutation.size(); i++) {
    if (permutation[i] == axis) {
      new_axis = i;
      break;
    }
  }
  NNADAPTER_CHECK_GE(new_axis, 0);
  return new_axis;
}

}  // namespace nnadapter
