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

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#include "utility/debug.h"
#include "utility/micros.h"
#include "utility/string.h"
#include "utility/utility.h"

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

NNADAPTER_EXPORT int64_t
GetOperandPrecisionDataLength(NNAdapterOperandPrecisionCode type) {
  switch (type) {
    case NNADAPTER_BOOL8:
    case NNADAPTER_INT8:
    case NNADAPTER_UINT8:
    case NNADAPTER_TENSOR_BOOL8:
    case NNADAPTER_TENSOR_INT8:
    case NNADAPTER_TENSOR_UINT8:
    case NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER:
    case NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL:
    case NNADAPTER_TENSOR_QUANT_UINT8_ASYMM_PER_LAYER:
      return 1;
    case NNADAPTER_INT16:
    case NNADAPTER_UINT16:
    case NNADAPTER_FLOAT16:
    case NNADAPTER_TENSOR_INT16:
    case NNADAPTER_TENSOR_UINT16:
    case NNADAPTER_TENSOR_FLOAT16:
      return 2;
    case NNADAPTER_INT32:
    case NNADAPTER_UINT32:
    case NNADAPTER_FLOAT32:
    case NNADAPTER_TENSOR_INT32:
    case NNADAPTER_TENSOR_UINT32:
    case NNADAPTER_TENSOR_FLOAT32:
    case NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_LAYER:
    case NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_CHANNEL:
    case NNADAPTER_TENSOR_QUANT_UINT32_ASYMM_PER_LAYER:
      return 4;
    case NNADAPTER_INT64:
    case NNADAPTER_UINT64:
    case NNADAPTER_FLOAT64:
    case NNADAPTER_TENSOR_INT64:
    case NNADAPTER_TENSOR_UINT64:
    case NNADAPTER_TENSOR_FLOAT64:
      return 8;
    default:
      NNADAPTER_LOG(ERROR) << "Failed to get the length of type("
                           << static_cast<int>(type) << ").";
      break;
  }
  return 0;
}

NNADAPTER_EXPORT int64_t
GetOperandTypeBufferLength(const NNAdapterOperandType& type) {
  auto production =
      ProductionOfDimensions(type.dimensions, type.dimension_count);
  return GetOperandPrecisionDataLength(type.precision) * production;
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

NNADAPTER_EXPORT std::vector<int32_t> MultiplyPermutation(
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

NNADAPTER_EXPORT void Symm2AsymmData(const int8_t* input_data,
                                     size_t input_data_count,
                                     int32_t zero_point,
                                     uint8_t* output_data) {
  int i = 0;
  int size = input_data_count;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  int16x8_t vzp_s16x8 = vdupq_n_s16(zero_point);
  for (; i < size - 7; i += 8) {
    int8x8_t vin_s8x8 = vld1_s8(input_data);
    int16x8_t vin_s16x8 = vmovl_s8(vin_s8x8);
    int16x8_t vout_s16x8 = vqaddq_s16(vin_s16x8, vzp_s16x8);
    uint8x8_t vout_u8x8 = vqmovun_s16(vout_s16x8);
    vst1_u8(output_data, vout_u8x8);
    input_data += 8;
    output_data += 8;
  }
#endif
  for (; i < size; i++) {
    *(output_data++) = static_cast<uint8_t>(std::min(
        std::max(static_cast<int16_t>(*(input_data++)) + zero_point, 0), 255));
  }
}

NNADAPTER_EXPORT void Asymm2SymmData(const uint8_t* input_data,
                                     size_t input_data_count,
                                     int32_t zero_point,
                                     int8_t* output_data) {
  int i = 0;
  int size = input_data_count;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  int16x8_t vzp_s16x8 = vdupq_n_s16(zero_point);
  for (; i < size - 7; i += 8) {
    uint8x8_t vin_u8x8 = vld1_u8(input_data);
    int16x8_t vin_s16x8 = vreinterpretq_s16_u16(vmovl_u8(vin_u8x8));
    int16x8_t vout_s16x8 = vqsubq_s16(vin_s16x8, vzp_s16x8);
    int8x8_t vout_s8x8 = vqmovn_s16(vout_s16x8);
    vst1_s8(output_data, vout_s8x8);
    input_data += 8;
    output_data += 8;
  }
#endif
  for (; i < size; i++) {
    *(output_data++) = static_cast<int8_t>(std::min(
        std::max(static_cast<int16_t>(*(input_data++)) - zero_point, -128),
        127));
  }
}

NNADAPTER_EXPORT int32_t
TransposeAxis(int32_t axis, const std::vector<int32_t>& permutation) {
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

NNADAPTER_EXPORT std::map<std::string, std::string> GetKeyValues(
    const char* properties,
    const std::string& delimiter,
    const std::string& assignment) {
  std::map<std::string, std::string> key_values;
  auto sections = string_split(properties, delimiter);
  for (auto section : sections) {
    auto tokens = string_split(section, assignment);
    NNADAPTER_CHECK_EQ(tokens.size(), 2);
    auto key = tokens[0];
    auto value = tokens[1];
    NNADAPTER_CHECK(!key.empty() && !value.empty());
    key_values[key] = value;
  }
  return key_values;
}

NNADAPTER_EXPORT uint32_t CRC32C(const uint8_t* buffer, size_t size) {
  const uint32_t polynomial = 0x82F63B78;
  uint32_t result = 0;
  for (size_t i = 0; i < size; i++) {
    result ^= buffer[i];
    for (int j = 0; j < 8; j++) {
      if (result & 1) {
        result = (result >> 1) ^ polynomial;
      } else {
        result >>= 1;
      }
    }
  }
  return result;
}

NNADAPTER_EXPORT bool ReadFile(const std::string& path,
                               std::vector<uint8_t>* buffer) {
  FILE* fp = fopen(path.c_str(), "rb");
  if (!fp) return false;
  fseek(fp, 0, SEEK_END);
  size_t size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  buffer->clear();
  buffer->resize(size);
  size_t offset = 0;
  char* ptr = reinterpret_cast<char*>(&(buffer->at(0)));
  while (offset < size) {
    size_t already_read = fread(ptr, 1, size - offset, fp);
    offset += already_read;
    ptr += already_read;
  }
  fclose(fp);
  return true;
}

NNADAPTER_EXPORT bool WriteFile(const std::string& path,
                                const std::vector<uint8_t>& buffer) {
  FILE* fp = fopen(path.c_str(), "wb");
  if (!fp) return false;
  size_t size = buffer.size();
  size_t offset = 0;
  const char* ptr = reinterpret_cast<const char*>(&(buffer.at(0)));
  while (offset < size) {
    size_t already_written = fwrite(ptr, 1, size - offset, fp);
    offset += already_written;
    ptr += already_written;
  }
  fclose(fp);
  return true;
}

}  // namespace nnadapter
